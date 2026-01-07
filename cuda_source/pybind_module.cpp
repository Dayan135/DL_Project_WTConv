#include <torch/extension.h>
#include <vector>
#include "cuda_kernels.h"

// ------------------------------------------------------------------
// C++ wrappers calling CUDA kernels (Atomic Ops)
// ------------------------------------------------------------------
torch::Tensor dwt_fwd_op(const torch::Tensor& input, float scale) {
    int N = input.size(0); int C = input.size(1); int H = input.size(2); int W = input.size(3);
    auto output = torch::empty({N, 4*C, H/2, W/2}, input.options());
    launch_dwt_forward(input.data_ptr<float>(), output.data_ptr<float>(), N, C, H, W, scale);
    return output;
}

torch::Tensor dwt_bwd_op(const torch::Tensor& grad_out, int H_orig, int W_orig, float scale) {
    // grad_out: N, 4C, H/2, W/2
    int N = grad_out.size(0); int C4 = grad_out.size(1); int C = C4/4;
    auto grad_in = torch::empty({N, C, H_orig, W_orig}, grad_out.options());
    launch_dwt_backward(grad_out.data_ptr<float>(), grad_in.data_ptr<float>(), N, C, H_orig, W_orig, scale);
    return grad_in;
}

torch::Tensor conv_fwd_op(const torch::Tensor& input, const torch::Tensor& weight, int stride, int pad) {
    int N = input.size(0); int C = input.size(1); int H = input.size(2); int W = input.size(3);
    int K = weight.size(2);
    int H_out = (H + 2*pad - K)/stride + 1;
    int W_out = (W + 2*pad - K)/stride + 1;
    auto output = torch::empty({N, C, H_out, W_out}, input.options());
    launch_conv_depthwise_fwd(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
                              N, C, H, W, K, stride, pad, H_out, W_out);
    return output;
}

// ... (Bind backward conv ops similarly) ...

torch::Tensor idwt_fwd_op(const torch::Tensor& input, float scale) {
    // input: N, 4C, H/2, W/2
    int N = input.size(0); int C4 = input.size(1); int C = C4/4; int H_sub = input.size(2); int W_sub = input.size(3);
    auto output = torch::empty({N, C, H_sub*2, W_sub*2}, input.options());
    launch_idwt_forward(input.data_ptr<float>(), output.data_ptr<float>(), N, C, H_sub*2, W_sub*2, scale);
    return output;
}

// ------------------------------------------------------------------
// MAIN: Multi-Level Forward (The Loop)
// ------------------------------------------------------------------
// Returns: {Output Tensor, List of Saved Tensors for Backward}
std::tuple<torch::Tensor, std::vector<torch::Tensor>> wtconv_forward_multilevel(
    torch::Tensor input, 
    std::vector<torch::Tensor> weights, 
    int stride, int pad, int groups, 
    float dwt_scale, float idwt_scale) 
{
    // Validate
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA");
    int levels = weights.size();
    
    std::vector<torch::Tensor> saved_tensors; // For backward
    torch::Tensor curr_ll = input;
    
    // 1. Downward Path (DWT + Conv)
    // We store: [Original_Shape_0, Processed_LL_0, Processed_High_0, ...]
    
    // We need to store high-freq results to combine later.
    // Structure of saved_tensors per level:
    // 0: Input Shape (Tensor)
    // 1: Processed Low (LL) - (Wait, this goes to next level)
    // 2: Processed High (LH, HL, HH)
    
    std::vector<torch::Tensor> processed_highs;
    std::vector<std::vector<int64_t>> shapes;

    for (int i = 0; i < levels; ++i) {
        shapes.push_back(curr_ll.sizes().vec());
        
        // A. DWT
        // Output: (N, 4C, H/2, W/2) Interleaved
        torch::Tensor dwt_out = dwt_fwd_op(curr_ll, dwt_scale);
        
        // B. Conv
        // Convolve ALL bands (4C) using depthwise kernel
        // Weights[i] shape: (4C, 1, K, K)
        torch::Tensor conv_out = conv_fwd_op(dwt_out, weights[i], stride, pad);
        
        // C. Split (LibTorch)
        // conv_out layout: interleaved [LL, LH, HL, HH]
        // We need to separate LL (for next level) from Highs (for reconstruction)
        // Reshape to (N, C, 4, H', W') to split
        int N = conv_out.size(0); 
        int C4 = conv_out.size(1); 
        int H_out = conv_out.size(2); 
        int W_out = conv_out.size(3);
        int C = C4/4;
        
        // Reshape: (N, 4*C, ...) -> (N, C, 4, ...)
        // Note: My kernel writes 4*c+0, 4*c+1... so this reshape is valid memory-wise
        auto view = conv_out.view({N, C, 4, H_out, W_out});
        
        // Extract LL (Band 0)
        torch::Tensor next_ll = view.slice(2, 0, 1).squeeze(2).contiguous();
        
        // Extract Highs (Bands 1, 2, 3)
        // We keep them interleaved or planar? 
        // IDWT kernel expects interleaved (N, 4C, ...).
        // So we keep conv_out intact, but we need to ZERO OUT the LL part later?
        // BGU Logic: curr_x_ll = x_ll_in_levels.pop() ...
        
        // Let's save the whole conv_out for this level, but we update the LL part later.
        processed_highs.push_back(conv_out); // This has the High bands we need
        
        curr_ll = next_ll; // Recurse
    }
    
    // 2. Upward Path (IDWT + Combine)
    // The BGU logic: 
    // curr_x_ll = 0 (initially)
    // Loop back:
    //   Pop processed_highs (contains LL_processed, LH, HL, HH)
    //   We REPLACE the LL part with (LL_processed + curr_x_ll_from_deeper)
    //   Then IDWT.
    
    torch::Tensor recon_ll = torch::zeros_like(curr_ll); // Start with 0 (or residual?)
    
    for (int i = levels - 1; i >= 0; --i) {
        torch::Tensor level_data = processed_highs[i]; // (N, 4C, H, W)
        
        // View to access bands
        int N = level_data.size(0);
        int C4 = level_data.size(1);
        int H = level_data.size(2);
        int W = level_data.size(3);
        int C = C4/4;
        
        auto view = level_data.view({N, C, 4, H, W});
        
        // Get the LL from this level
        auto level_ll = view.slice(2, 0, 1).squeeze(2);
        
        // Combine: New LL = This_Level_LL + Recon_LL_From_Below
        // Note: Shapes must match.
        auto combined_ll = level_ll + recon_ll;
        
        // Pack back into interleaved format for IDWT
        // We need to construct a tensor (N, 4C, H, W) where band 0 is combined_ll
        // and bands 1-3 are from level_data.
        
        // Slicing and copying
        // Clone level_data so we don't modify saved list if we needed it later
        auto input_to_idwt = level_data.clone();
        auto input_view = input_to_idwt.view({N, C, 4, H, W});
        
        // Overwrite Band 0
        input_view.select(2, 0).copy_(combined_ll);
        
        // IDWT
        recon_ll = idwt_fwd_op(input_to_idwt, idwt_scale);
        
        // BGU: Crop if padding was added? My kernels assume perfect sizes for now.
    }
    
    return std::make_tuple(recon_ll, saved_tensors);
}

// Wrapper for Python
torch::Tensor wtconv_forward_py(torch::Tensor input, std::vector<torch::Tensor> weights, 
                                int stride, int pad, int groups, 
                                float dwt_scale, float idwt_scale) {
    auto result = wtconv_forward_multilevel(input, weights, stride, pad, groups, dwt_scale, idwt_scale);
    return std::get<0>(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wtconv_forward", &wtconv_forward_py, "Multi-Level Forward");
    // Expose atomic ops for debugging/custom autograd if needed
    m.def("dwt_forward", &dwt_fwd_op);
    m.def("idwt_forward", &idwt_fwd_op);
    m.def("conv_forward", &conv_fwd_op);
}