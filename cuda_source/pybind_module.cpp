#include <torch/extension.h>
#include <vector>
#include <tuple>
#include "cuda_kernel.h"

// ==================================================================
// Wrappers for Naive Kernels
// ==================================================================

torch::Tensor dwt_forward(const torch::Tensor& input, float scale) {
    int N = input.size(0); int C = input.size(1); int H = input.size(2); int W = input.size(3);
    auto output = torch::empty({N, 4*C, H/2, W/2}, input.options());
    launch_dwt_forward(input.data_ptr<float>(), output.data_ptr<float>(), N, C, H, W, scale);
    return output;
}

torch::Tensor dwt_backward(const torch::Tensor& grad_output, float scale) {
    // grad_output: (N, 4C, H/2, W/2) -> grad_input: (N, C, H, W)
    int N = grad_output.size(0); int C4 = grad_output.size(1); int C = C4/4;
    int H_sub = grad_output.size(2); int W_sub = grad_output.size(3);
    auto grad_input = torch::empty({N, C, H_sub*2, W_sub*2}, grad_output.options());
    launch_dwt_backward(grad_output.data_ptr<float>(), grad_input.data_ptr<float>(), N, C, H_sub*2, W_sub*2, scale);
    return grad_input;
}

torch::Tensor idwt_forward(const torch::Tensor& input, float scale) {
    int N = input.size(0); int C4 = input.size(1); int C = C4/4; 
    int H_sub = input.size(2); int W_sub = input.size(3);
    auto output = torch::empty({N, C, H_sub*2, W_sub*2}, input.options());
    launch_idwt_forward(input.data_ptr<float>(), output.data_ptr<float>(), N, C, H_sub*2, W_sub*2, scale);
    return output;
}

torch::Tensor idwt_backward(const torch::Tensor& grad_output, float scale) {
    // grad_output: (N, C, H, W) -> grad_input: (N, 4C, H/2, W/2)
    int N = grad_output.size(0); int C = grad_output.size(1);
    int H = grad_output.size(2); int W = grad_output.size(3);
    auto grad_input = torch::empty({N, 4*C, H/2, W/2}, grad_output.options());
    launch_idwt_backward(grad_output.data_ptr<float>(), grad_input.data_ptr<float>(), N, C, H, W, scale);
    return grad_input;
}

torch::Tensor conv_forward(const torch::Tensor& input, const torch::Tensor& weight, int stride, int pad) {
    int N = input.size(0); int C = input.size(1); int H = input.size(2); int W = input.size(3);
    int K = weight.size(2);
    int H_out = (H + 2*pad - K)/stride + 1;
    int W_out = (W + 2*pad - K)/stride + 1;
    auto output = torch::empty({N, C, H_out, W_out}, input.options());
    launch_conv_depthwise_fwd(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
                              N, C, H, W, K, stride, pad, H_out, W_out);
    return output;
}

std::tuple<torch::Tensor, torch::Tensor> conv_backward(const torch::Tensor& grad_output, 
                                                       const torch::Tensor& input, 
                                                       const torch::Tensor& weight, 
                                                       int stride, int pad) {
    int N = input.size(0); int C = input.size(1); int H = input.size(2); int W = input.size(3);
    int K = weight.size(2);
    int H_out = grad_output.size(2); int W_out = grad_output.size(3);
    
    auto grad_input = torch::empty_like(input);
    auto grad_weight = torch::empty_like(weight);
    
    launch_conv_depthwise_bwd_in(grad_output.data_ptr<float>(), weight.data_ptr<float>(), grad_input.data_ptr<float>(),
                                 N, C, H, W, K, stride, pad, H_out, W_out);
                                 
    launch_conv_depthwise_bwd_w(grad_output.data_ptr<float>(), input.data_ptr<float>(), grad_weight.data_ptr<float>(),
                                N, C, H_out, W_out, K, stride, pad, H, W);
                                
    return std::make_tuple(grad_input, grad_weight);
}

// ==================================================================
// Naive Logic
// ==================================================================

// 1. Forward (Pure)
torch::Tensor wtconv_forward_naive(torch::Tensor input, std::vector<torch::Tensor> weights, 
                                   int stride, int pad, int groups, 
                                   float dwt_scale, float idwt_scale) {
    int levels = weights.size();
    std::vector<torch::Tensor> processed_highs;
    torch::Tensor curr_ll = input;
    
    for (int i=0; i<levels; ++i) {
        auto coeffs = dwt_forward(curr_ll, dwt_scale);
        
        int N = coeffs.size(0); int C4 = coeffs.size(1); int C = C4/4;
        int H = coeffs.size(2); int W = coeffs.size(3);
        
        auto coeffs_view = coeffs.view({N, C, 4, H, W});
        auto next_ll = coeffs_view.select(2, 0).clone(); 
        
        auto conv_out = conv_forward(coeffs, weights[i], stride, pad);
        processed_highs.push_back(conv_out);
        
        curr_ll = next_ll;
    }
    
    torch::Tensor recon = torch::zeros_like(curr_ll);
    
    for (int i=levels-1; i>=0; --i) {
        torch::Tensor level_res = processed_highs[i];
        int N = level_res.size(0); int C = level_res.size(1) / 4;
        int H = level_res.size(2); int W = level_res.size(3);
        
        auto view = level_res.view({N, C, 4, H, W});
        auto level_ll = view.select(2, 0);
        auto combined = level_ll + recon;
        
        auto idwt_in = level_res.clone();
        idwt_in.view({N, C, 4, H, W}).select(2, 0).copy_(combined);
        
        recon = idwt_forward(idwt_in, idwt_scale);
    }
    
    return recon;
}

// 2. Forward + Save (For Training)
std::tuple<torch::Tensor, std::vector<torch::Tensor>> wtconv_forward_save(
    torch::Tensor input, std::vector<torch::Tensor> weights, 
    int stride, int pad, int groups, float dwt_scale, float idwt_scale) 
{
    int levels = weights.size();
    std::vector<torch::Tensor> saved_coeffs; 
    std::vector<torch::Tensor> processed_highs;
    torch::Tensor curr_ll = input;
    
    for (int i=0; i<levels; ++i) {
        auto coeffs = dwt_forward(curr_ll, dwt_scale);
        saved_coeffs.push_back(coeffs); 
        
        int N = coeffs.size(0); int C4 = coeffs.size(1); int C = C4/4;
        int H = coeffs.size(2); int W = coeffs.size(3);
        auto coeffs_view = coeffs.view({N, C, 4, H, W});
        auto next_ll = coeffs_view.select(2, 0).clone();
        
        auto conv_out = conv_forward(coeffs, weights[i], stride, pad);
        processed_highs.push_back(conv_out);
        
        curr_ll = next_ll;
    }
    
    torch::Tensor recon = torch::zeros_like(curr_ll);
    for (int i=levels-1; i>=0; --i) {
        torch::Tensor level_res = processed_highs[i];
        int N = level_res.size(0); int C = level_res.size(1) / 4;
        int H = level_res.size(2); int W = level_res.size(3);
        auto view = level_res.view({N, C, 4, H, W});
        auto level_ll = view.select(2, 0);
        auto combined = level_ll + recon;
        auto idwt_in = level_res.clone();
        idwt_in.view({N, C, 4, H, W}).select(2, 0).copy_(combined);
        recon = idwt_forward(idwt_in, idwt_scale);
    }
    
    return std::make_tuple(recon, saved_coeffs);
}

// 3. Backward (Fixed: Two-Pass Strategy)
std::tuple<torch::Tensor, std::vector<torch::Tensor>> wtconv_backward(
    std::vector<torch::Tensor> saved_coeffs, torch::Tensor grad_out, 
    std::vector<torch::Tensor> weights, int groups) 
{
    int levels = weights.size();
    std::vector<torch::Tensor> grad_weights(levels);
    std::vector<torch::Tensor> grad_conv_inputs(levels); // Gradient w.r.t saved_coeffs[i]
    
    float dwt_scale = 0.5;
    float idwt_scale = 0.5;
    
    // --- PASS 1: Reconstruction Backward (Top -> Deep) ---
    // Start with the output gradient (64x64) and propagate down to get 
    // gradients for each convolution layer.
    
    torch::Tensor curr_grad_recon = grad_out; 
    
    for (int i = 0; i < levels; ++i) {
        // A. IDWT Backward
        // Takes larger image (e.g. 64x64) -> splits to smaller bands (e.g. 32x32)
        auto grad_combined = idwt_backward(curr_grad_recon, idwt_scale);
        
        // B. Conv Backward
        torch::Tensor input_to_conv = saved_coeffs[i];
        torch::Tensor w = weights[i];
        
        auto res = conv_backward(grad_combined, input_to_conv, w, 1, w.size(2)/2);
        grad_conv_inputs[i] = std::get<0>(res); 
        grad_weights[i] = std::get<1>(res);
        
        // C. Extract Gradient for Next Level (Skip Connection)
        // The LL band of grad_combined is the gradient w.r.t the 'recon' accumulator 
        // passed from the deeper level.
        int N = grad_combined.size(0); int C = grad_combined.size(1)/4;
        int H = grad_combined.size(2); int W = grad_combined.size(3);
        auto view = grad_combined.view({N, C, 4, H, W});
        curr_grad_recon = view.select(2, 0).clone(); // Save for next iter
    }
    
    // --- PASS 2: Decomposition Backward (Deep -> Top) ---
    // Propagate gradients back up the decomposition path.
    
    torch::Tensor grad_from_deep = torch::zeros_like(curr_grad_recon); 
    // For i=levels-1, the LL band wasn't used, so gradient is 0.
    
    for (int i = levels - 1; i >= 0; --i) {
        // 1. Accumulate
        torch::Tensor total_grad_coeffs = grad_conv_inputs[i];
        
        if (i < levels - 1) {
            // Add gradient from deeper level to LL band
            int N = total_grad_coeffs.size(0); int C = total_grad_coeffs.size(1)/4;
            int H = total_grad_coeffs.size(2); int W = total_grad_coeffs.size(3);
            auto view = total_grad_coeffs.view({N, C, 4, H, W});
            auto ll_band = view.select(2, 0);
            ll_band.add_(grad_from_deep);
        }
        
        // 2. DWT Backward
        grad_from_deep = dwt_backward(total_grad_coeffs, dwt_scale);
    }
    
    return std::make_tuple(grad_from_deep, grad_weights);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wtconv_forward", &wtconv_forward_naive);
    m.def("wtconv_forward_save", &wtconv_forward_save);
    m.def("wtconv_backward", &wtconv_backward);
}