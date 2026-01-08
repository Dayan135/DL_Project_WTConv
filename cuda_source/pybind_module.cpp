#include <torch/extension.h>
#include <vector>
#include <tuple>
#include "cuda_kernel.h"

// ================================================================
// 0) Small helpers
// ================================================================
static inline bool any_requires_grad(const std::vector<torch::Tensor>& xs) {
    for (const auto& t : xs) {
        if (t.defined() && t.requires_grad()) return true;
    }
    return false;
}

static inline void check_cuda_f32_contig(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// ================================================================
// 1) Atomic ops wrappers (your code, with minimal checks)
// ================================================================
torch::Tensor dwt_fwd_op(const torch::Tensor& input, float scale) {
    check_cuda_f32_contig(input, "dwt_fwd_op input");
    int N = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    TORCH_CHECK((H % 2) == 0 && (W % 2) == 0, "DWT expects even H,W");
    auto output = torch::empty({N, 4*C, H/2, W/2}, input.options());
    launch_dwt_forward(input.data_ptr<float>(), output.data_ptr<float>(), N, C, H, W, scale);
    return output;
}

torch::Tensor dwt_bwd_op(const torch::Tensor& grad_out, int H_orig, int W_orig, float scale) {
    check_cuda_f32_contig(grad_out, "dwt_bwd_op grad_out");
    int N = grad_out.size(0);
    int C4 = grad_out.size(1);
    TORCH_CHECK(C4 % 4 == 0, "dwt_bwd_op: channels must be multiple of 4");
    int C = C4 / 4;
    auto grad_in = torch::empty({N, C, H_orig, W_orig}, grad_out.options());
    launch_dwt_backward(grad_out.data_ptr<float>(), grad_in.data_ptr<float>(), N, C, H_orig, W_orig, scale);
    return grad_in;
}

torch::Tensor idwt_fwd_op(const torch::Tensor& input, float scale) {
    check_cuda_f32_contig(input, "idwt_fwd_op input");
    int N = input.size(0);
    int C4 = input.size(1);
    TORCH_CHECK(C4 % 4 == 0, "idwt_fwd_op: channels must be multiple of 4");
    int C = C4/4;
    int H_sub = input.size(2), W_sub = input.size(3);
    auto output = torch::empty({N, C, H_sub*2, W_sub*2}, input.options());
    launch_idwt_forward(input.data_ptr<float>(), output.data_ptr<float>(), N, C, H_sub*2, W_sub*2, scale);
    return output;
}

// IDWT backward is DWT on grad_output (you already have launch_idwt_backward)
torch::Tensor idwt_bwd_op(const torch::Tensor& grad_out, float scale) {
    check_cuda_f32_contig(grad_out, "idwt_bwd_op grad_out");
    int N = grad_out.size(0), C = grad_out.size(1), H = grad_out.size(2), W = grad_out.size(3);
    TORCH_CHECK((H % 2) == 0 && (W % 2) == 0, "idwt_bwd_op expects even H,W");
    auto grad_in = torch::empty({N, 4*C, H/2, W/2}, grad_out.options());
    launch_idwt_backward(grad_out.data_ptr<float>(), grad_in.data_ptr<float>(), N, C, H, W, scale);
    return grad_in;
}

torch::Tensor conv_fwd_op(const torch::Tensor& input, const torch::Tensor& weight, int stride, int pad) {
    check_cuda_f32_contig(input, "conv_fwd_op input");
    check_cuda_f32_contig(weight, "conv_fwd_op weight");

    int N = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    int K = weight.size(2);
    int H_out = (H + 2*pad - K)/stride + 1;
    int W_out = (W + 2*pad - K)/stride + 1;

    auto output = torch::empty({N, C, H_out, W_out}, input.options());
    launch_conv_depthwise_fwd(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
                              N, C, H, W, K, stride, pad, H_out, W_out);
    return output;
}

torch::Tensor conv_bwd_in_op(const torch::Tensor& grad_out,
                            const torch::Tensor& weight,
                            int H_in, int W_in, int stride, int pad) {
    check_cuda_f32_contig(grad_out, "conv_bwd_in_op grad_out");
    check_cuda_f32_contig(weight, "conv_bwd_in_op weight");

    int N = grad_out.size(0);
    int C = grad_out.size(1);
    int H_out = grad_out.size(2);
    int W_out = grad_out.size(3);
    int K = weight.size(2);

    auto grad_in = torch::empty({N, C, H_in, W_in}, grad_out.options());
    launch_conv_depthwise_bwd_in(grad_out.data_ptr<float>(), weight.data_ptr<float>(), grad_in.data_ptr<float>(),
                                 N, C, H_in, W_in, K, stride, pad, H_out, W_out);
    return grad_in;
}

torch::Tensor conv_bwd_w_op(const torch::Tensor& grad_out,
                           const torch::Tensor& input,
                           int K, int stride, int pad) {
    check_cuda_f32_contig(grad_out, "conv_bwd_w_op grad_out");
    check_cuda_f32_contig(input, "conv_bwd_w_op input");

    int N = grad_out.size(0);
    int C = grad_out.size(1);
    int H_out = grad_out.size(2);
    int W_out = grad_out.size(3);
    int H_in = input.size(2);
    int W_in = input.size(3);

    auto grad_w = torch::zeros({C, 1, K, K}, grad_out.options());
    launch_conv_depthwise_bwd_w(grad_out.data_ptr<float>(), input.data_ptr<float>(), grad_w.data_ptr<float>(),
                                N, C, H_out, W_out, K, stride, pad, H_in, W_in);
    return grad_w;
}

// ================================================================
// 2) Forward core
// ================================================================
// saved layout (vector<tensor>):
// [0]  levels_cpu (int64 scalar, CPU)
// [1]  stride_cpu (int64 scalar, CPU)
// [2]  pad_cpu    (int64 scalar, CPU)
// [3]  dwt_scale_cpu  (float32 scalar, CPU)
// [4]  idwt_scale_cpu (float32 scalar, CPU)
//
// Then per level i=0..L-1 (in forward order):
//   shape_i_cpu  : int64 tensor [2] = {H_ll, W_ll}  (CPU)
//   dwt_out_i    : CUDA tensor (N,4C,H/2,W/2)        (CUDA)  (needed if weight grad)
//   conv_out_i   : CUDA tensor (N,4C,H',W')          (CUDA)  (needed for reconstruction backward)
//
// Notes:
// - We always save shape + conv_out_i if backward needed.
// - We save dwt_out_i only if any weight requires grad.
std::tuple<torch::Tensor, std::vector<torch::Tensor>>
wtconv_forward_multilevel_save(torch::Tensor input,
                               std::vector<torch::Tensor> weights,
                               int stride, int pad, int groups,
                               float dwt_scale, float idwt_scale,
                               bool save_for_backward) {
    check_cuda_f32_contig(input, "wtconv input");
    TORCH_CHECK(!weights.empty(), "weights list is empty");
    int levels = (int)weights.size();

    for (int i = 0; i < levels; ++i) {
        check_cuda_f32_contig(weights[i], "weight[i]");
    }

    bool want_wgrad = save_for_backward && any_requires_grad(weights);

    std::vector<torch::Tensor> saved;
    if (save_for_backward) {
        saved.reserve(5 + levels * (2 + (want_wgrad ? 1 : 0)));
        saved.push_back(torch::tensor({(int64_t)levels}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)));
        saved.push_back(torch::tensor({(int64_t)stride}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)));
        saved.push_back(torch::tensor({(int64_t)pad}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)));
        saved.push_back(torch::tensor({(float)dwt_scale}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)));
        saved.push_back(torch::tensor({(float)idwt_scale}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)));
    }

    torch::Tensor curr_ll = input;
    std::vector<torch::Tensor> conv_outs;
    conv_outs.reserve(levels);

    // ---- Downward path
    for (int i = 0; i < levels; ++i) {
        int64_t H_ll = curr_ll.size(2);
        int64_t W_ll = curr_ll.size(3);

        if (save_for_backward) {
            auto shape_cpu = torch::tensor({H_ll, W_ll}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
            saved.push_back(shape_cpu);
        }

        torch::Tensor dwt_out = dwt_fwd_op(curr_ll, dwt_scale);

        // Depthwise conv over (4C) channels
        torch::Tensor conv_out = conv_fwd_op(dwt_out, weights[i], stride, pad);

        if (save_for_backward) {
            if (want_wgrad) saved.push_back(dwt_out);
            saved.push_back(conv_out);
        }

        conv_outs.push_back(conv_out);

        // next_ll = band0(conv_out)
        int64_t N = conv_out.size(0);
        int64_t C4 = conv_out.size(1);
        int64_t Hc = conv_out.size(2);
        int64_t Wc = conv_out.size(3);
        TORCH_CHECK(C4 % 4 == 0, "conv_out channels must be multiple of 4");
        int64_t C = C4 / 4;

        auto view = conv_out.view({N, C, 4, Hc, Wc});
        torch::Tensor next_ll = view.select(2, 0).contiguous();

        curr_ll = next_ll;
    }

    // ---- Upward path (reconstruct)
    torch::Tensor recon_ll = torch::zeros_like(curr_ll);

    for (int i = levels - 1; i >= 0; --i) {
        torch::Tensor level_data = conv_outs[i]; // (N,4C,H,W)
        int64_t N = level_data.size(0);
        int64_t C4 = level_data.size(1);
        int64_t H = level_data.size(2);
        int64_t W = level_data.size(3);
        int64_t C = C4 / 4;

        auto view = level_data.view({N, C, 4, H, W});
        torch::Tensor level_ll = view.select(2, 0); // (N,C,H,W)
        torch::Tensor combined_ll = level_ll + recon_ll;

        // clone for packing band0
        torch::Tensor input_to_idwt = level_data.clone();
        auto input_view = input_to_idwt.view({N, C, 4, H, W});
        input_view.select(2, 0).copy_(combined_ll);

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

    return {recon_ll, saved};
}

// ================================================================
// 3) Public forward
// ================================================================
torch::Tensor wtconv_forward_py(torch::Tensor input,
                               std::vector<torch::Tensor> weights,
                               int stride, int pad, int groups,
                               float dwt_scale, float idwt_scale) {
    // Torch-like behavior: if no grad is enabled OR nothing requires grad, take fast no-save path.
    bool need_backward = torch::GradMode::is_enabled() &&
                         (input.requires_grad() || any_requires_grad(weights));

    // For plain forward API, we never return saved anyway; so we can always run without saving.
    // This guarantees benchmark matches your expected "no_grad" behavior.
    auto out = wtconv_forward_multilevel_save(input, weights, stride, pad, groups, dwt_scale, idwt_scale, /*save=*/false);
    return std::get<0>(out);
}

// Forward that explicitly returns saved tensors (used by Python autograd wrapper)
std::tuple<torch::Tensor, std::vector<torch::Tensor>>
wtconv_forward_save_py(torch::Tensor input,
                       std::vector<torch::Tensor> weights,
                       int stride, int pad, int groups,
                       float dwt_scale, float idwt_scale) {
    // When called from autograd.Function.forward(), GradMode is disabled by PyTorch,
    // but we still need to save if any input requires_grad.
    // So we check requires_grad instead of GradMode.
    bool need_backward = input.requires_grad() || any_requires_grad(weights);

    if (!need_backward) {
        // Match torch semantics: if no backward is needed, don't waste memory saving.
        auto out = wtconv_forward_multilevel_save(input, weights, stride, pad, groups, dwt_scale, idwt_scale, /*save=*/false);
        return {std::get<0>(out), std::vector<torch::Tensor>()};
    }

    auto out = wtconv_forward_multilevel_save(input, weights, stride, pad, groups, dwt_scale, idwt_scale, /*save=*/true);
    return out;
}

// ================================================================
// 4) Backward from saved
// ================================================================
std::tuple<torch::Tensor, std::vector<torch::Tensor>>
wtconv_backward_py(const std::vector<torch::Tensor>& saved,
                   torch::Tensor grad_out,
                   std::vector<torch::Tensor> weights,
                   int groups) {
    check_cuda_f32_contig(grad_out, "grad_out");
    TORCH_CHECK(!weights.empty(), "weights list is empty");
    int levels = (int)weights.size();
    for (int i = 0; i < levels; ++i) check_cuda_f32_contig(weights[i], "weight[i]");

    // If saved is empty => forward did not save => no backward possible
    TORCH_CHECK(!saved.empty(), "saved is empty (forward did not save). Run wtconv_forward_save under grad.");

    // Read header
    TORCH_CHECK(saved.size() >= 5, "saved is corrupted");
    int L = (int)saved[0].to(torch::kCPU).item<int64_t>();
    TORCH_CHECK(L == levels, "saved levels != weights.size()");
    int stride = (int)saved[1].to(torch::kCPU).item<int64_t>();
    int pad    = (int)saved[2].to(torch::kCPU).item<int64_t>();
    float dwt_scale  = saved[3].to(torch::kCPU).item<float>();
    float idwt_scale = saved[4].to(torch::kCPU).item<float>();

    // Determine if dwt_out was saved (weight grad)
    // Layout after header:
    // per level: shape_cpu, [dwt_out if present], conv_out
    // We detect by counting expected size.
    // expected_no_wgrad = 5 + L*(1 + 1) = 5 + 2L
    // expected_wgrad    = 5 + L*(1 + 1 + 1) = 5 + 3L
    int64_t expected_no_wgrad = 5 + 2LL*L;
    int64_t expected_wgrad    = 5 + 3LL*L;

    bool has_dwt_saved = false;
    if ((int64_t)saved.size() == expected_wgrad) has_dwt_saved = true;
    else if ((int64_t)saved.size() == expected_no_wgrad) has_dwt_saved = false;
    else TORCH_CHECK(false, "saved size mismatch: got ", saved.size(),
                     ", expected ", expected_no_wgrad, " or ", expected_wgrad);

    // Unpack per-level saved
    std::vector<torch::Tensor> shape_cpu(L);
    std::vector<torch::Tensor> dwt_saved(L);
    std::vector<torch::Tensor> conv_saved(L);

    int idx = 5;
    for (int i = 0; i < L; ++i) {
        shape_cpu[i] = saved[idx++]; // CPU [H,W]
        if (has_dwt_saved) dwt_saved[i] = saved[idx++]; // CUDA dwt_out
        conv_saved[i]  = saved[idx++]; // CUDA conv_out
    }

    // ------------------------------------------------------------
    // Phase A: backprop through reconstruction (upward path)
    // Produce grad_conv_out[i] with same shape as conv_saved[i]
    // ------------------------------------------------------------
    std::vector<torch::Tensor> grad_conv_out(L);
    torch::Tensor grad_recon_ll = grad_out;

    for (int i = L - 1; i >= 0; --i) {
        torch::Tensor level_data = conv_saved[i]; // forward conv_out for shape
        int64_t N = level_data.size(0);
        int64_t C4 = level_data.size(1);
        int64_t H = level_data.size(2);
        int64_t W = level_data.size(3);
        int64_t C = C4 / 4;

        // grad wrt input_to_idwt (interleaved)
        torch::Tensor g_input = idwt_bwd_op(grad_recon_ll, idwt_scale); // (N,4C,H,W)

        // band0 gradient is grad wrt combined_ll
        auto g_view = g_input.view({N, C, 4, H, W});
        torch::Tensor g_combined_ll = g_view.select(2, 0); // (N,C,H,W)

        // combined_ll = level_ll + recon_prev
        // so grad flows:
        //   grad_level_ll += g_combined_ll
        //   grad_recon_prev = g_combined_ll
        torch::Tensor grad_recon_prev = g_combined_ll;

        // Build grad for level_data:
        // start from g_input (already has grads for highs)
        // and add grad to band0 (level_ll path)
        torch::Tensor g_level_data = g_input.clone();
        auto g_level_view = g_level_data.view({N, C, 4, H, W});
        g_level_view.select(2, 0).add_(g_combined_ll);

        grad_conv_out[i] = g_level_data;
        grad_recon_ll = grad_recon_prev;
    }

    // ------------------------------------------------------------
    // Phase B: backprop through down path (conv + dwt)
    // ------------------------------------------------------------
    std::vector<torch::Tensor> grad_weights(L);
    torch::Tensor grad_next_ll; // gradient into next_ll from deeper level

    for (int i = L - 1; i >= 0; --i) {
        torch::Tensor g_conv = grad_conv_out[i];

        // Add gradient coming from deeper down-path into band0 (since next_ll came from band0)
        if (grad_next_ll.defined()) {
            torch::Tensor level_data = conv_saved[i];
            int64_t N = level_data.size(0);
            int64_t C4 = level_data.size(1);
            int64_t H = level_data.size(2);
            int64_t W = level_data.size(3);
            int64_t C = C4 / 4;

            auto g_view = g_conv.view({N, C, 4, H, W});
            g_view.select(2, 0).add_(grad_next_ll);
        }

        // Conv backward
        int K = (int)weights[i].size(2);

        // Need input size for conv_bwd_in = dwt_out spatial size
        // If dwt_saved exists use it, else we must recompute (slow).
        torch::Tensor dwt_in;
        if (has_dwt_saved) {
            dwt_in = dwt_saved[i];
        } else {
            // recompute dwt_out by reconstructing curr_ll is complex.
            // So for correctness: require dwt_saved if any weight requires grad OR if input grad needs it.
            // Practically: always run forward_save with has_dwt_saved when you want backward.
            TORCH_CHECK(false, "dwt_out was not saved; backward cannot proceed without recomputation path.");
        }

        int H_in = (int)dwt_in.size(2);
        int W_in = (int)dwt_in.size(3);

        torch::Tensor g_dwt = conv_bwd_in_op(g_conv, weights[i], H_in, W_in, stride, pad);
        torch::Tensor g_w   = conv_bwd_w_op(g_conv, dwt_in, K, stride, pad);
        grad_weights[i] = g_w;

        // DWT backward to get grad wrt curr_ll
        auto hw = shape_cpu[i].to(torch::kCPU);
        int H_ll = (int)hw[0].item<int64_t>();
        int W_ll = (int)hw[1].item<int64_t>();
        torch::Tensor g_ll = dwt_bwd_op(g_dwt, H_ll, W_ll, dwt_scale);

        grad_next_ll = g_ll; // propagate to previous level
    }

    torch::Tensor grad_input = grad_next_ll;
    return {grad_input, grad_weights};
}

// ================================================================
// 5) PyBind
// ================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Forward-only (fast, no saving)
    m.def("wtconv_forward", &wtconv_forward_py, "WTConv multi-level forward (no save)");

    // Forward with save (Torch-like: returns empty saved if no backward is needed)
    m.def("wtconv_forward_save", &wtconv_forward_save_py, "WTConv forward + saved tensors");

    // Backward using saved tensors
    m.def("wtconv_backward", &wtconv_backward_py, "WTConv backward from saved");

    // Debug atomic ops
    m.def("dwt_forward", &dwt_fwd_op);
    m.def("idwt_forward", &idwt_fwd_op);
    m.def("conv_forward", &conv_fwd_op);
}
