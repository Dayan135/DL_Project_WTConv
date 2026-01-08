#include <torch/extension.h>
#include <vector>
#include "cpp_kernel.h"

// ------------------------------------------------------------------
// Atomic Wrappers (Calling Raw C++ Kernels)
// ------------------------------------------------------------------

torch::Tensor dwt_fwd_op(const torch::Tensor& input, float scale) {
    int N = input.size(0); int C = input.size(1); int H = input.size(2); int W = input.size(3);
    auto output = torch::empty({N, 4*C, H/2, W/2}, input.options());
    
    // Call raw C++ kernel
    haar_dwt_2d(input.data_ptr<float>(), output.data_ptr<float>(), N, C, H, W, scale);
    return output;
}

torch::Tensor idwt_fwd_op(const torch::Tensor& input, float scale) {
    int N = input.size(0); int C4 = input.size(1); int C = C4/4; 
    int H_sub = input.size(2); int W_sub = input.size(3);
    auto output = torch::empty({N, C, H_sub*2, W_sub*2}, input.options());
    
    haar_idwt_2d(input.data_ptr<float>(), output.data_ptr<float>(), N, C, H_sub*2, W_sub*2, scale);
    return output;
}

torch::Tensor conv_fwd_op(const torch::Tensor& input, const torch::Tensor& weight, int stride, int pad, int groups) {
    int N = input.size(0); int C = input.size(1); int H = input.size(2); int W = input.size(3);
    int Cout_4 = weight.size(0); int K = weight.size(2);
    int Cout = Cout_4; // Weight is already [Cout, Cin/G, K, K]

    int H_out = (H + 2*pad - K)/stride + 1;
    int W_out = (W + 2*pad - K)/stride + 1;
    
    auto output = torch::empty({N, Cout, H_out, W_out}, input.options());
    
    conv2d_forward_impl(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
                        N, C, Cout, H, W, K, stride, pad, groups);
    return output;
}

// ------------------------------------------------------------------
// MAIN: Multi-Level Forward (CPU)
// ------------------------------------------------------------------
std::tuple<torch::Tensor, std::vector<torch::Tensor>> wtconv_forward_multilevel(
    torch::Tensor input, 
    std::vector<torch::Tensor> weights, 
    int stride, int pad, int groups, 
    float dwt_scale, float idwt_scale) 
{
    // Ensure CPU
    TORCH_CHECK(!input.is_cuda(), "Input must be CPU Tensor for cpp_module");

    int levels = weights.size();
    std::vector<torch::Tensor> saved_tensors;
    std::vector<torch::Tensor> processed_highs;

    torch::Tensor curr_ll = input;
    
    // --- Downward Path ---
    for (int i = 0; i < levels; ++i) {
        // 1. DWT
        torch::Tensor dwt_out = dwt_fwd_op(curr_ll, dwt_scale);
        
        // 2. Extract Raw LL for NEXT level
        // View as (N, C, 4, H, W)
        int N = dwt_out.size(0); int C4 = dwt_out.size(1); int C = C4/4;
        int H_out = dwt_out.size(2); int W_out = dwt_out.size(3);
        
        auto dwt_view = dwt_out.view({N, C, 4, H_out, W_out});
        // Important: Clone to ensure it owns memory separate from dwt_out
        torch::Tensor next_ll = dwt_view.select(2, 0).clone();

        // 3. Conv
        torch::Tensor conv_out = conv_fwd_op(dwt_out, weights[i], stride, pad, groups);
        processed_highs.push_back(conv_out);
        
        // 4. Update
        curr_ll = next_ll;
    }
    
    // --- Upward Path ---
    torch::Tensor recon_ll = torch::zeros_like(curr_ll);
    
    for (int i = levels - 1; i >= 0; --i) {
        torch::Tensor level_data = processed_highs[i];
        
        int N = level_data.size(0); int C4 = level_data.size(1); int C = C4/4;
        int H = level_data.size(2); int W = level_data.size(3);

        auto view = level_data.view({N, C, 4, H, W});
        auto level_ll = view.select(2, 0); 
        
        // Combine
        auto combined_ll = level_ll + recon_ll;
        
        // Pack into IDWT input
        auto input_to_idwt = level_data.clone();
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
    m.def("wtconv_forward", &wtconv_forward_py, "Multi-Level Forward (CPU)");
}