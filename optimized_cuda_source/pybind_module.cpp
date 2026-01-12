#include <torch/extension.h>
#include <vector>
#include <tuple>
#include "cuda_kernel.h"

// ------------------------------------------------------------------
// HELPERS (Atomic Ops)
// ------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> fused_wtconv_op(const torch::Tensor& input, const torch::Tensor& weight, float scale) {
    int N = input.size(0); int C = input.size(1); int H = input.size(2); int W = input.size(3);
    int K = weight.size(2);
    auto output = torch::empty({N, 4*C, H/2, W/2}, input.options());
    auto next_ll = torch::empty({N, C, H/2, W/2}, input.options());
    launch_fused_wtconv_fwd(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), next_ll.data_ptr<float>(), N, C, H/2, W/2, K, scale);
    return std::make_tuple(output, next_ll);
}

torch::Tensor fused_idwt_add_op(const torch::Tensor& coeffs, const torch::optional<torch::Tensor>& deep_recon, float scale) {
    int N = coeffs.size(0); int C = coeffs.size(1)/4; int H = coeffs.size(2); int W = coeffs.size(3);
    auto output = torch::empty({N, C, H*2, W*2}, coeffs.options());
    const float* rp = deep_recon.has_value() ? deep_recon.value().data_ptr<float>() : nullptr;
    launch_fused_idwt_add(coeffs.data_ptr<float>(), rp, output.data_ptr<float>(), N, C, H, W, scale);
    return output;
}

// ------------------------------------------------------------------
// 1. UNIFIED FORWARD (Handles both Inference and Training)
// ------------------------------------------------------------------
std::tuple<torch::Tensor, std::vector<torch::Tensor>> wtconv_forward(
    torch::Tensor input, 
    std::vector<torch::Tensor> weights, 
    int stride, int pad, int groups, 
    float dwt_scale, float idwt_scale,
    bool training = false) // <--- Default to inference mode
{
    int levels = weights.size();
    
    // We always return a vector, but it stays empty if training=false
    std::vector<torch::Tensor> saved_inputs; 
    if (training) {
        saved_inputs.reserve(levels);
    }

    std::vector<torch::Tensor> processed_highs; 
    processed_highs.reserve(levels);
    
    torch::Tensor curr_ll = input;
    
    // --- Downward Path ---
    for (int i = 0; i < levels; ++i) {
        if (training) {
            saved_inputs.push_back(curr_ll); // Save input for backward re-materialization
        }
        
        auto result = fused_wtconv_op(curr_ll, weights[i], dwt_scale);
        processed_highs.push_back(std::get<0>(result));
        curr_ll = std::get<1>(result); 
    }
    
    // --- Upward Path ---
    torch::Tensor recon;
    for (int i = levels - 1; i >= 0; --i) {
        // If it's the deepest level, deep_recon is null (start of reconstruction)
        // Otherwise, it uses the 'recon' from the previous iteration
        auto deep_recon = (i == levels - 1) ? torch::nullopt : std::make_optional(recon);
        recon = fused_idwt_add_op(processed_highs[i], deep_recon, idwt_scale);
    }
    
    return std::make_tuple(recon, saved_inputs);
}

// ------------------------------------------------------------------
// 2. BACKWARD (Same as before)
// ------------------------------------------------------------------
std::tuple<torch::Tensor, std::vector<torch::Tensor>> wtconv_backward(
    std::vector<torch::Tensor> saved_inputs, torch::Tensor grad_out, 
    std::vector<torch::Tensor> weights, int groups) 
{
    int levels = weights.size();
    torch::Tensor curr_grad_recon = grad_out; 
    std::vector<torch::Tensor> grad_weights(levels);
    float scale = 0.5;

    for (int i = levels - 1; i >= 0; --i) {
        torch::Tensor input_img = saved_inputs[i];
        int N = input_img.size(0); int C = input_img.size(1);
        int H = input_img.size(2); int W = input_img.size(3);
        int H_sub = H/2; int W_sub = W/2;
        
        // 1. Re-materialize
        auto dwt_coeffs = torch::empty({N, 4*C, H_sub, W_sub}, input_img.options());
        launch_fused_dwt_split(input_img.data_ptr<float>(), dwt_coeffs.data_ptr<float>(), nullptr, N, C, H_sub, W_sub, scale);
        
        // 2. Split Gradient
        auto grad_conv_out = torch::empty({N, 4*C, H_sub, W_sub}, curr_grad_recon.options());
        launch_fused_dwt_split(curr_grad_recon.data_ptr<float>(), grad_conv_out.data_ptr<float>(), nullptr, N, C, H_sub, W_sub, scale);

        // 3. Weight Gradients
        torch::Tensor w = weights[i];
        int K = w.size(2); int pad = K/2;
        auto grad_w = torch::empty_like(w);
        launch_conv_depthwise_bwd_w(grad_conv_out.data_ptr<float>(), dwt_coeffs.data_ptr<float>(), grad_w.data_ptr<float>(), N, 4*C, H_sub, W_sub, K, 1, pad, H_sub, W_sub);
        grad_weights[i] = grad_w;

        // 4. Input Grads
        auto grad_input = torch::empty({N, C, H, W}, curr_grad_recon.options());
        launch_fused_conv_bwd_idwt(grad_conv_out.data_ptr<float>(), w.data_ptr<float>(), grad_input.data_ptr<float>(), N, C, H_sub, W_sub, K, scale);
        
        curr_grad_recon = grad_input;
    }
    return std::make_tuple(curr_grad_recon, grad_weights);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Only two functions needed now!
    m.def("wtconv_forward", &wtconv_forward, "Unified Forward (Inference/Train)", 
          py::arg("input"), py::arg("weights"), py::arg("stride"), py::arg("pad"), 
          py::arg("groups"), py::arg("dwt_scale"), py::arg("idwt_scale"), 
          py::arg("training") = false);
          
    m.def("wtconv_backward", &wtconv_backward, "Fused Backward");
}