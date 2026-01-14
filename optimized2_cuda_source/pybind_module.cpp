#include <torch/extension.h>
#include <vector>
#include <tuple>
#include "cuda_kernel.h"

// Helpers
std::tuple<torch::Tensor, torch::Tensor> fused_wtconv_op(const torch::Tensor& input,
                                                         const torch::Tensor& weight,
                                                         float scale) {
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int K = weight.size(2);

    auto output  = torch::empty({N, 4 * C, H / 2, W / 2}, input.options());
    auto next_ll = torch::empty({N, C, H / 2, W / 2}, input.options());

    launch_fused_wtconv_fwd(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        next_ll.data_ptr<float>(),
        N, C, H / 2, W / 2, K, scale
    );

    return std::make_tuple(output, next_ll);
}

torch::Tensor fused_idwt_add_op(const torch::Tensor& coeffs,
                                const torch::optional<torch::Tensor>& deep_recon,
                                float scale) {
    int N = coeffs.size(0);
    int C = coeffs.size(1) / 4;
    int H = coeffs.size(2);
    int W = coeffs.size(3);

    auto output = torch::empty({N, C, H * 2, W * 2}, coeffs.options());
    const float* rp = deep_recon.has_value() ? deep_recon.value().data_ptr<float>() : nullptr;

    launch_fused_idwt_add(
        coeffs.data_ptr<float>(),
        rp,
        output.data_ptr<float>(),
        N, C, H, W, scale
    );

    return output;
}

// Forward
std::tuple<torch::Tensor, std::vector<torch::Tensor>> wtconv_forward(
    torch::Tensor input,
    std::vector<torch::Tensor> weights,
    int stride, int pad, int groups,
    float dwt_scale, float idwt_scale,
    bool training = false
) {
    (void)stride; (void)pad; (void)groups;

    int levels = (int)weights.size();
    std::vector<torch::Tensor> saved_inputs;
    if (training) saved_inputs.reserve(levels);

    std::vector<torch::Tensor> processed_highs;
    processed_highs.reserve(levels);

    torch::Tensor curr_ll = input;

    for (int i = 0; i < levels; ++i) {
        if (training) saved_inputs.push_back(curr_ll);

        auto result = fused_wtconv_op(curr_ll, weights[i], dwt_scale);
        processed_highs.push_back(std::get<0>(result));
        curr_ll = std::get<1>(result);
    }

    torch::Tensor recon;
    for (int i = levels - 1; i >= 0; --i) {
        auto deep_recon = (i == levels - 1) ? torch::nullopt : std::make_optional(recon);
        recon = fused_idwt_add_op(processed_highs[i], deep_recon, idwt_scale);
    }

    return std::make_tuple(recon, saved_inputs);
}

// Backward (Two-Pass)
std::tuple<torch::Tensor, std::vector<torch::Tensor>> wtconv_backward(
    std::vector<torch::Tensor> saved_inputs,
    torch::Tensor grad_out,
    std::vector<torch::Tensor> weights,
    int groups,
    float dwt_scale, float idwt_scale
) {
    (void)groups;

    int levels = (int)weights.size();
    std::vector<torch::Tensor> grad_weights(levels);
    std::vector<torch::Tensor> grad_conv_outs(levels);

    // Pass 1: Top -> Deep (Decompose Reconstruction Gradients)
    torch::Tensor g = grad_out;
    for (int i = 0; i < levels; ++i) {
        int N = g.size(0);
        int C = g.size(1);
        int H = g.size(2) / 2;
        int W = g.size(3) / 2;

        auto gc = torch::empty({N, 4 * C, H, W}, g.options());
        auto gr = torch::empty({N, C, H, W}, g.options());

        launch_fused_dwt_split(
            g.data_ptr<float>(),
            gc.data_ptr<float>(),
            gr.data_ptr<float>(),
            N, C, H, W, idwt_scale
        );

        grad_conv_outs[i] = gc;
        g = gr;
    }

    torch::Tensor grad_from_deeper = torch::Tensor();

    // Pass 2: Deep -> Top (Compute Input/Weight Gradients)
    for (int i = levels - 1; i >= 0; --i) {
        torch::Tensor input_img = saved_inputs[i];
        torch::Tensor w = weights[i];

        int N = input_img.size(0);
        int C = input_img.size(1);
        int H = input_img.size(2);
        int W = input_img.size(3);

        int H_sub = H / 2;
        int W_sub = W / 2;

        // Rematerialize DWT
        auto dwt_coeffs = torch::empty({N, 4 * C, H_sub, W_sub}, input_img.options());
        launch_fused_dwt_split(
            input_img.data_ptr<float>(),
            dwt_coeffs.data_ptr<float>(),
            nullptr,
            N, C, H_sub, W_sub, dwt_scale
        );

        // Weight Grad
        int K = w.size(2);
        auto grad_w = torch::empty_like(w);
        launch_conv_depthwise_bwd_w(
            grad_conv_outs[i].data_ptr<float>(),
            dwt_coeffs.data_ptr<float>(),
            grad_w.data_ptr<float>(),
            N, 4 * C, H_sub, W_sub,
            K, 1, K / 2, H_sub, W_sub
        );
        grad_weights[i] = grad_w;

        // Input Grad
        auto grad_input = torch::empty({N, C, H, W}, input_img.options());
        const float* g_deep_ptr = (i == levels - 1) ? nullptr : grad_from_deeper.data_ptr<float>();

        launch_fused_conv_bwd_idwt(
            grad_conv_outs[i].data_ptr<float>(),
            g_deep_ptr,
            w.data_ptr<float>(),
            grad_input.data_ptr<float>(),
            N, C, H_sub, W_sub, K, dwt_scale
        );

        grad_from_deeper = grad_input;
    }

    return std::make_tuple(grad_from_deeper, grad_weights);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wtconv_forward", &wtconv_forward, "Unified Forward",
          py::arg("input"), py::arg("weights"),
          py::arg("stride"), py::arg("pad"), py::arg("groups"),
          py::arg("dwt_scale"), py::arg("idwt_scale"),
          py::arg("training") = false);

    m.def("wtconv_backward", &wtconv_backward, "Fused Backward",
          py::arg("saved_inputs"), py::arg("grad_out"),
          py::arg("weights"), py::arg("groups"),
          py::arg("dwt_scale"), py::arg("idwt_scale"));
}