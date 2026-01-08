#include <torch/extension.h>
#include <vector>
#include <tuple>
#include "cuda_kernel.h"

// ------------------------------------------------------------------
// 1. FUSED CONV OP (Downward)
// ------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> fused_wtconv_op(
    const torch::Tensor& input, 
    const torch::Tensor& weight, 
    float scale) 
{
    int N = input.size(0); 
    int C = input.size(1); 
    int H = input.size(2); 
    int W = input.size(3);
    int K = weight.size(2);
    
    // Output 1: Convolved Bands (N, 4C, H/2, W/2)
    auto output = torch::empty({N, 4*C, H/2, W/2}, input.options());
    
    // Output 2: Next Level LL (N, C, H/2, W/2)
    auto next_ll = torch::empty({N, C, H/2, W/2}, input.options());

    launch_fused_wtconv_fwd(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        next_ll.data_ptr<float>(),
        N, C, H, W, K, scale
    );

    return std::make_tuple(output, next_ll);
}

// ------------------------------------------------------------------
// 2. FUSED IDWT + ADD OP (Upward)
// ------------------------------------------------------------------
torch::Tensor fused_idwt_add_op(const torch::Tensor& coeffs, 
                                const torch::optional<torch::Tensor>& deep_recon, 
                                float scale) {
    int N = coeffs.size(0);
    int C4 = coeffs.size(1);
    int C = C4 / 4;
    int H = coeffs.size(2);
    int W = coeffs.size(3);
    
    // Output is double the size
    auto output = torch::empty({N, C, H*2, W*2}, coeffs.options());
    
    const float* recon_ptr = nullptr;
    if (deep_recon.has_value()) {
        recon_ptr = deep_recon.value().data_ptr<float>();
    }

    launch_fused_idwt_add(
        coeffs.data_ptr<float>(),
        recon_ptr,
        output.data_ptr<float>(),
        N, C, H, W, scale
    );
    
    return output;
}

// ------------------------------------------------------------------
// 3. MAIN PYTORCH FORWARD
// ------------------------------------------------------------------
torch::Tensor wtconv_forward_fused_py(torch::Tensor input, std::vector<torch::Tensor> weights, 
                                      int stride, int pad, int groups, 
                                      float dwt_scale, float idwt_scale) {
    
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA");
    int levels = weights.size();
    
    // Store forward results
    std::vector<torch::Tensor> processed_highs; 
    
    // --- Downward Path (Fused) ---
    torch::Tensor curr_ll = input;
    for (int i = 0; i < levels; ++i) {
        auto result = fused_wtconv_op(curr_ll, weights[i], dwt_scale);
        processed_highs.push_back(std::get<0>(result));
        curr_ll = std::get<1>(result); 
    }
    
    // --- Upward Path (Fused) ---
    // At the deepest level, there is no "previous" reconstruction to add.
    torch::Tensor recon_ll;
    
    for (int i = levels - 1; i >= 0; --i) {
        torch::Tensor level_data = processed_highs[i]; 
        
        if (i == levels - 1) {
             // Deepest level: pass nullopt
             recon_ll = fused_idwt_add_op(level_data, torch::nullopt, idwt_scale);
        } else {
             // Higher levels: pass previous result
             recon_ll = fused_idwt_add_op(level_data, recon_ll, idwt_scale);
        }
    }
    
    return recon_ll;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wtconv_forward", &wtconv_forward_fused_py, "Fused Multi-Level");
    m.def("fused_wtconv_op", &fused_wtconv_op);
    m.def("fused_idwt_add_op", &fused_idwt_add_op);
}