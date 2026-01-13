#include <torch/extension.h>
#include <vector>
#include <tuple>
#include "cpp_kernel.h"

// Helper to ensure contiguity
torch::Tensor make_contiguous(const torch::Tensor& t) {
    return t.contiguous();
}

// ------------------------------------------------------------------
// Atomic Wrappers
// ------------------------------------------------------------------
torch::Tensor dwt_fwd_op(const torch::Tensor& input, float scale) {
    auto in_c = make_contiguous(input);
    int N = in_c.size(0); int C = in_c.size(1); int H = in_c.size(2); int W = in_c.size(3);
    auto output = torch::empty({N, 4*C, H/2, W/2}, in_c.options());
    haar_dwt_2d(in_c.data_ptr<float>(), output.data_ptr<float>(), N, C, H, W, scale);
    return output;
}

torch::Tensor idwt_fwd_op(const torch::Tensor& input, float scale) {
    auto in_c = make_contiguous(input);
    int N = in_c.size(0); int C4 = in_c.size(1); int C = C4/4; 
    int H_sub = in_c.size(2); int W_sub = in_c.size(3);
    auto output = torch::empty({N, C, H_sub*2, W_sub*2}, in_c.options());
    haar_idwt_2d(in_c.data_ptr<float>(), output.data_ptr<float>(), N, C, H_sub*2, W_sub*2, scale);
    return output;
}

// ------------------------------------------------------------------
// UNIFIED FORWARD
// ------------------------------------------------------------------
std::tuple<torch::Tensor, std::vector<torch::Tensor>> wtconv_forward(
    torch::Tensor input, std::vector<torch::Tensor> weights, 
    int stride, int pad, int groups, float dwt_scale, float idwt_scale,
    bool training = false) 
{
    TORCH_CHECK(!input.is_cuda(), "Input must be CPU");
    
    int levels = weights.size();
    std::vector<torch::Tensor> saved; 
    std::vector<torch::Tensor> processed_highs;
    torch::Tensor curr_ll = make_contiguous(input); // Ensure contiguous start
    
    // Down
    for (int i = 0; i < levels; ++i) {
        torch::Tensor dwt_out = dwt_fwd_op(curr_ll, dwt_scale);
        if (training) saved.push_back(dwt_out);
        
        int N = dwt_out.size(0); int C = dwt_out.size(1)/4; int H = dwt_out.size(2); int W = dwt_out.size(3);
        auto dwt_view = dwt_out.view({N, C, 4, H, W});
        torch::Tensor next_ll = dwt_view.select(2, 0).clone(); // Clone ensures contiguous

        int Cout = weights[i].size(0); int K = weights[i].size(2);
        int H_out = (H + 2*pad - K)/stride + 1; int W_out = (W + 2*pad - K)/stride + 1;
        auto conv_out = torch::empty({N, Cout, H_out, W_out}, input.options());
        
        auto w_contig = make_contiguous(weights[i]);
        conv2d_forward_impl(dwt_out.data_ptr<float>(), w_contig.data_ptr<float>(), conv_out.data_ptr<float>(),
                            N, C*4, Cout, H, W, K, stride, pad, groups);
        processed_highs.push_back(conv_out);
        curr_ll = next_ll;
    }
    
    // Up
    torch::Tensor recon_ll = torch::zeros_like(curr_ll);
    for (int i = levels - 1; i >= 0; --i) {
        torch::Tensor level_data = processed_highs[i];
        int N = level_data.size(0); int C = level_data.size(1)/4; int H = level_data.size(2); int W = level_data.size(3);

        auto view = level_data.view({N, C, 4, H, W});
        auto level_ll = view.select(2, 0); 
        auto combined_ll = level_ll + recon_ll;
        
        // Critical: Make contiguous copy before IDWT
        auto input_to_idwt = level_data.clone();
        input_to_idwt.view({N, C, 4, H, W}).select(2, 0).copy_(combined_ll);
        
        recon_ll = idwt_fwd_op(input_to_idwt, idwt_scale);
    }
    
    return std::make_tuple(recon_ll, saved);
}

// ------------------------------------------------------------------
// BACKWARD
// ------------------------------------------------------------------
std::tuple<torch::Tensor, std::vector<torch::Tensor>> wtconv_backward(
    std::vector<torch::Tensor> saved, torch::Tensor grad_out, 
    std::vector<torch::Tensor> weights, int groups) 
{
    int levels = weights.size();
    std::vector<torch::Tensor> grad_weights(levels);
    float dwt_scale = 0.5; float idwt_scale = 0.5;

    // Pass 1: Top -> Deep
    std::vector<torch::Tensor> grad_conv_outs(levels);
    torch::Tensor g = make_contiguous(grad_out);
    
    for(int i=0; i<levels; ++i) {
        torch::Tensor grad_split = dwt_fwd_op(g, idwt_scale); 
        grad_conv_outs[i] = grad_split;
        int N = grad_split.size(0); int C = grad_split.size(1)/4; int H = grad_split.size(2); int W = grad_split.size(3);
        // Prepare contiguous gradient for next level
        g = grad_split.view({N, C, 4, H, W}).select(2, 0).clone(); 
    }

    // Pass 2: Deep -> Top
    torch::Tensor grad_from_deep = torch::Tensor(); 
    
    for (int i = levels - 1; i >= 0; --i) {
        torch::Tensor input_to_conv = make_contiguous(saved[i]); 
        torch::Tensor w = make_contiguous(weights[i]);
        
        // Add Skip Connection Gradient
        if (i < levels - 1) {
             int N = grad_conv_outs[i].size(0); int C = grad_conv_outs[i].size(1)/4; 
             int H = grad_conv_outs[i].size(2); int W = grad_conv_outs[i].size(3);
             grad_conv_outs[i].view({N, C, 4, H, W}).select(2, 0).add_(grad_from_deep);
        }
        
        int N = input_to_conv.size(0); int Cin = input_to_conv.size(1); 
        int H = input_to_conv.size(2); int W = input_to_conv.size(3);
        int K = w.size(2); int Cout = grad_conv_outs[i].size(1);
        
        auto grad_input = torch::zeros_like(input_to_conv);
        auto grad_w = torch::zeros_like(w);
        
        // Force contiguous memory for the convolution gradient input
        auto grad_conv_out_c = make_contiguous(grad_conv_outs[i]);

        conv2d_backward_impl(grad_conv_out_c.data_ptr<float>(), input_to_conv.data_ptr<float>(), w.data_ptr<float>(),
                             grad_input.data_ptr<float>(), grad_w.data_ptr<float>(),
                             N, Cin, Cout, H, W, K, 1, K/2, groups);
                             
        grad_weights[i] = grad_w;
        grad_from_deep = idwt_fwd_op(grad_input, dwt_scale);
    }
    
    return std::make_tuple(grad_from_deep, grad_weights);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wtconv_forward", &wtconv_forward, "Unified Forward", 
          py::arg("input"), py::arg("weights"), py::arg("stride"), py::arg("pad"),
          py::arg("groups"), py::arg("dwt_scale"), py::arg("idwt_scale"), 
          py::arg("training") = false);

    m.def("wtconv_backward", &wtconv_backward, "Backward", 
          py::arg("saved_inputs"), py::arg("grad_out"), py::arg("weights"), py::arg("groups"));
}