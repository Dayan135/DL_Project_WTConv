// NOTE: Per-level right/bottom padding is required before each DWT to ensure even H/W for the DWT kernel at every level. After each IDWT, crop to the original (pre-pad) shape for that level. This ensures output matches reference for odd H/W and multi-level. The saved structure now includes per-level shape metadata for correct backward.

#include <torch/extension.h>
#include <vector>
#include <tuple>
#include "cuda_kernel.h"

#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <c10/util/irange.h>


// Pad tensor on right/bottom by (pad_h, pad_w) with zeros (NCHW), explicit 4D indexing
namespace idx = torch::indexing;
torch::Tensor pad_right_bottom(const torch::Tensor& input, int pad_h, int pad_w) {
    if (pad_h == 0 && pad_w == 0) return input;
    TORCH_CHECK(input.dim() == 4, "pad_right_bottom expects NCHW");
    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t H = input.size(2);
    const int64_t W = input.size(3);
    auto out = torch::zeros({N, C, H + pad_h, W + pad_w}, input.options());
    out.index_put_({idx::Slice(), idx::Slice(), idx::Slice(0, H), idx::Slice(0, W)}, input);
    return out;
}

// Crop tensor to (H, W) spatial size (NCHW), explicit 4D indexing and contiguous
torch::Tensor crop_to_shape(const torch::Tensor& input, int64_t H, int64_t W) {
    TORCH_CHECK(input.dim() == 4, "crop_to_shape expects NCHW");
    return input.index({idx::Slice(), idx::Slice(), idx::Slice(0, H), idx::Slice(0, W)}).contiguous();
}

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
    std::vector<std::vector<int64_t>> saved_shapes; // {orig_H, orig_W, pad_h, pad_w}
    if (training) saved_inputs.reserve(levels);
    saved_shapes.reserve(levels);

    std::vector<torch::Tensor> processed_highs;
    processed_highs.reserve(levels);

    torch::Tensor curr_ll = input;

    for (int i = 0; i < levels; ++i) {
        // Store original shape for this level
        int64_t H = curr_ll.size(2);
        int64_t W = curr_ll.size(3);
        int64_t pad_h = H % 2;
        int64_t pad_w = W % 2;
        saved_shapes.push_back({H, W, pad_h, pad_w});
        if (training) saved_inputs.push_back(curr_ll);
        // Pad right/bottom if needed
        if (pad_h || pad_w) curr_ll = pad_right_bottom(curr_ll, pad_h, pad_w);
        // Assert even H/W before DWT kernel
        TORCH_CHECK(curr_ll.size(2) % 2 == 0 && curr_ll.size(3) % 2 == 0,
            "DWT input must be even; got H=", curr_ll.size(2), " W=", curr_ll.size(3));
        auto result = fused_wtconv_op(curr_ll, weights[i], dwt_scale);
        processed_highs.push_back(std::get<0>(result));
        curr_ll = std::get<1>(result);
    }

    torch::Tensor recon;
    for (int i = levels - 1; i >= 0; --i) {
        torch::optional<torch::Tensor> deep_recon =
            (i == levels - 1) ? torch::nullopt : std::make_optional(recon.contiguous());
        recon = fused_idwt_add_op(processed_highs[i], deep_recon, idwt_scale);
        // Crop to original shape for this level
        int H = saved_shapes[i][0];
        int W = saved_shapes[i][1];
        recon = crop_to_shape(recon, H, W);
    }

    // Save shape metadata as a tensor for backward

    auto shape_meta = torch::empty({levels, 4}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    auto acc = shape_meta.accessor<int64_t, 2>();
    for (int i = 0; i < levels; ++i) {
        acc[i][0] = saved_shapes[i][0];
        acc[i][1] = saved_shapes[i][1];
        acc[i][2] = saved_shapes[i][2];
        acc[i][3] = saved_shapes[i][3];
    }

    // Return saved_inputs and shape_meta for backward
    std::vector<torch::Tensor> saved;
    if (training) saved = saved_inputs;
    saved.push_back(shape_meta);
    return std::make_tuple(recon, saved);
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

    // Extract shape metadata from saved_inputs (last tensor)

    torch::Tensor shape_meta = saved_inputs.back();
    saved_inputs.pop_back();
    TORCH_CHECK(shape_meta.device().is_cpu(), "shape_meta must be CPU");
    auto meta = shape_meta.accessor<int64_t, 2>();

    // Pass 1: Top -> Deep (Decompose Reconstruction Gradients)
    torch::Tensor g = grad_out;

    for (int i = 0; i < levels; ++i) {
        // Crop grad to the padded shape for this level before DWT split
        int64_t H = meta[i][0];
        int64_t W = meta[i][1];
        int64_t pad_h = meta[i][2];
        int64_t pad_w = meta[i][3];
        int64_t Hpad = H + pad_h;
        int64_t Wpad = W + pad_w;
        if (pad_h || pad_w) g = pad_right_bottom(g, pad_h, pad_w);
        TORCH_CHECK(g.size(2) % 2 == 0 && g.size(3) % 2 == 0,
            "Backward DWT input must be even; got H=", g.size(2), " W=", g.size(3));
        int N = g.size(0);
        int C = g.size(1);
        int H_sub = Hpad / 2;
        int W_sub = Wpad / 2;

        auto gc = torch::empty({N, 4 * C, H_sub, W_sub}, g.options());
        auto gr = torch::empty({N, C, H_sub, W_sub}, g.options());
        // Always provide a dummy tensor for grad_prev_ll
        auto dummy_ll = torch::empty({N, C, H_sub, W_sub}, g.options());
        launch_fused_dwt_split(
            g.data_ptr<float>(),
            gc.data_ptr<float>(),
            gr.data_ptr<float>(),
            N, C, H_sub, W_sub, idwt_scale
        );

        grad_conv_outs[i] = gc;
        g = gr;
    }

    torch::Tensor grad_from_deeper = torch::Tensor();

    // Pass 2: Deep -> Top (Compute Input/Weight Gradients)
    for (int i = levels - 1; i >= 0; --i) {
        torch::Tensor input_img = saved_inputs[i];
        torch::Tensor w = weights[i];
        int64_t H = meta[i][0];
        int64_t W = meta[i][1];
        int64_t pad_h = meta[i][2];
        int64_t pad_w = meta[i][3];
        int64_t Hpad = H + pad_h;
        int64_t Wpad = W + pad_w;
        // Pad input_img to padded shape before DWT
        if (pad_h || pad_w) input_img = pad_right_bottom(input_img, pad_h, pad_w);
        TORCH_CHECK(input_img.size(2) % 2 == 0 && input_img.size(3) % 2 == 0,
            "Backward DWT input_img must be even; got H=", input_img.size(2), " W=", input_img.size(3));
        int N = input_img.size(0);
        int C = input_img.size(1);
        int H_sub = Hpad / 2;
        int W_sub = Wpad / 2;

        // Rematerialize DWT
        auto dwt_coeffs = torch::empty({N, 4 * C, H_sub, W_sub}, input_img.options());
        auto dummy_ll = torch::empty({N, C, H_sub, W_sub}, input_img.options());
        launch_fused_dwt_split(
            input_img.data_ptr<float>(),
            dwt_coeffs.data_ptr<float>(),
            dummy_ll.data_ptr<float>(),
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
        auto grad_input = torch::empty({N, C, Hpad, Wpad}, input_img.options());
        const float* g_deep_ptr = (i == levels - 1) ? nullptr : grad_from_deeper.data_ptr<float>();

        launch_fused_conv_bwd_idwt(
            grad_conv_outs[i].data_ptr<float>(),
            g_deep_ptr,
            w.data_ptr<float>(),
            grad_input.data_ptr<float>(),
            N, C, H_sub, W_sub, K, dwt_scale
        );
        // Crop grad_input to original shape for this level
        grad_input = crop_to_shape(grad_input, H, W);
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