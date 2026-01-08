#include <torch/extension.h>
#include <vector>
#include "cuda_kernel.h"

// ------------------------------------------------------------------
// Atomic Operations (Unchanged)
// ------------------------------------------------------------------

torch::Tensor dwt_fwd_op(const torch::Tensor& input, float scale) {
    int N = input.size(0); int C = input.size(1); int H = input.size(2); int W = input.size(3);
    auto output = torch::empty({N, 4*C, H/2, W/2}, input.options());
    launch_dwt_forward(input.data_ptr<float>(), output.data_ptr<float>(), N, C, H, W, scale);
    return output;
}

torch::Tensor idwt_fwd_op(const torch::Tensor& input, float scale) {
    int N = input.size(0); int C4 = input.size(1); int C = C4/4; int H_sub = input.size(2); int W_sub = input.size(3);
    auto output = torch::empty({N, C, H_sub*2, W_sub*2}, input.options());
    launch_idwt_forward(input.data_ptr<float>(), output.data_ptr<float>(), N, C, H_sub*2, W_sub*2, scale);
    return output;
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

// ------------------------------------------------------------------
// MAIN: Multi-Level Forward (Corrected Logic)
// ------------------------------------------------------------------
std::tuple<torch::Tensor, std::vector<torch::Tensor>> wtconv_forward_multilevel(
    torch::Tensor input, 
    std::vector<torch::Tensor> weights, 
    int stride, int pad, int groups, 
    float dwt_scale, float idwt_scale) 
{
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA");
    int levels = weights.size();
    
    std::vector<torch::Tensor> saved_tensors;
    std::vector<torch::Tensor> processed_highs; // Stores CONVOLVED outputs

    torch::Tensor curr_ll = input;
    
    // --- Downward Path ---
    for (int i = 0; i < levels; ++i) {
        // 1. DWT (Decomposition)
        // Output: (N, 4C, H/2, W/2)
        torch::Tensor dwt_out = dwt_fwd_op(curr_ll, dwt_scale);
        
        // 2. Extract Raw LL for NEXT level [FIXED]
        // We must extract this from 'dwt_out' BEFORE convolution
        int N = dwt_out.size(0); 
        int C4 = dwt_out.size(1); 
        int C = C4/4;
        int H_out = dwt_out.size(2); 
        int W_out = dwt_out.size(3);
        
        // View as (N, C, 4, H, W) to separate bands
        auto dwt_view = dwt_out.view({N, C, 4, H_out, W_out});
        
        // Extract Band 0 (LL). Clone to own memory for next iteration.
        torch::Tensor next_ll = dwt_view.select(2, 0).clone();

        // 3. Conv (Transformation)
        // Convolve all bands (LL, LH, HL, HH)
        torch::Tensor conv_out = conv_fwd_op(dwt_out, weights[i], stride, pad);
        
        // Save CONVOLVED output for reconstruction
        processed_highs.push_back(conv_out);
        
        // 4. Update Loop Variable
        curr_ll = next_ll;
    }
    
    // --- Upward Path (Reconstruction) ---
    torch::Tensor recon_ll = torch::zeros_like(curr_ll);
    
    for (int i = levels - 1; i >= 0; --i) {
        torch::Tensor level_data = processed_highs[i]; // This is (N, 4C, H, W) CONVOLVED
        
        int N = level_data.size(0);
        int C4 = level_data.size(1);
        int C = C4/4;
        int H = level_data.size(2);
        int W = level_data.size(3);

        // View to separate bands
        auto view = level_data.view({N, C, 4, H, W});
        
        // Extract the CONVOLVED LL band from this level
        auto level_ll = view.select(2, 0); 
        
        // Combine: Convolved_LL (Current) + Reconstructed_LL (From Deeper)
        auto combined_ll = level_ll + recon_ll;
        
        // Prepare IDWT Input
        // We clone level_data to avoid in-place modification of saved history
        auto input_to_idwt = level_data.clone();
        
        // Overwrite the LL band with the combined result
        input_to_idwt.view({N, C, 4, H, W}).select(2, 0).copy_(combined_ll);
        
        // IDWT
        recon_ll = idwt_fwd_op(input_to_idwt, idwt_scale);
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
    m.def("dwt_forward", &dwt_fwd_op);
    m.def("idwt_forward", &idwt_fwd_op);
    m.def("conv_forward", &conv_fwd_op);
}