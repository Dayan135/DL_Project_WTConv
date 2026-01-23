# tests/test_equivalence.py
#
# Goal:
#   Proper correctness tests (forward + backward) against Reference WTConv2d.
#   Covers:
#     - even + odd spatial sizes (e.g., 33x35) to catch padding/cropping bugs
#     - core mode (wavelet path only, base branch disabled, wavelet_scale=1)
#     - optional full mode (if/when your CUDA wrapper implements base branch + scales)
#
# Usage:
#   pytest -q tests/test_equivalence.py
#   pytest -q tests/test_equivalence.py -k core
#   pytest -q tests/test_equivalence.py -k odd
#
import os
import sys
import pytest
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_REPO_PATH = os.path.join(HERE, "Reference")

# Make sure Reference is importable
sys.path.insert(0, REF_REPO_PATH)

# Extensions paths (compiled .so/.pyd live here)
sys.path.append(os.path.join(HERE, "cuda_source"))
sys.path.append(os.path.join(HERE, "optimized_cuda_source"))
sys.path.append(os.path.join(HERE, "optimized2_cuda_source"))

# --- Optional CUDA modules ---
try:
    import cuda_module
except ImportError:
    cuda_module = None

try:
    import optimized_cuda_module
except ImportError:
    optimized_cuda_module = None

try:
    import optimized2_cuda_module
except ImportError:
    optimized2_cuda_module = None


DWT_SCALE = 0.5
IDWT_SCALE = 0.5

# =============================================================================
# Reference fixture
# =============================================================================

@pytest.fixture(scope="session")
def wtconv_class():
    try:
        from wtconv.wtconv2d import WTConv2d
        return WTConv2d
    except Exception as e:
        pytest.fail(f"Could not import Reference WTConv2d: {e}")


def _make_reference_model(wtconv_class, C, K, levels, device, mode: str):
    """
    mode:
      - "core": disable base branch, set wavelet_scale=1
      - "full": do NOT override scales; test real WTConv2d behavior
    """
    m = wtconv_class(C, C, kernel_size=K, wt_levels=levels, wt_type="db1").to(device)
    m.eval()

    if mode == "core":
        # Disable base branch output:
        # x_out = base_scale(base_conv(x)) + x_tag  -> make base term 0
        m.base_scale.weight.data.fill_(0.0)
        if m.base_scale.bias is not None:
            m.base_scale.bias.data.fill_(0.0)

        # Neutralize wavelet gain (so CUDA doesn't need to model the learned scaling)
        for s in m.wavelet_scale:
            s.weight.data.fill_(1.0)

    elif mode == "full":
        # Keep exactly as Reference initialized (wavelet_scale init=0.1 etc.)
        pass
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return m


# =============================================================================
# Configs: include odd dims to catch padding/crop mismatches
# =============================================================================

@pytest.fixture(params=[
    # even sizes
    (2, 4, 32, 32, 5, 1),
    (2, 4, 32, 32, 5, 2),
    (1, 8, 64, 64, 3, 1),
    (1, 8, 64, 64, 7, 1),

    # odd sizes (critical)
    (1, 4, 33, 35, 5, 1),
    (1, 4, 33, 35, 5, 2),
], ids=[
    "even_B2_C4_32x32_K5_L1",
    "even_B2_C4_32x32_K5_L2",
    "even_B1_C8_64x64_K3_L1",
    "even_B1_C8_64x64_K7_L1",
    "odd_B1_C4_33x35_K5_L1",
    "odd_B1_C4_33x35_K5_L2",
])
def cfg(request):
    return request.param


# =============================================================================
# Helpers: weights formatting + module calling (baseline vs opt vs opt2)
# =============================================================================

def _weights_for_baseline(ref_layer, levels):
    # Reference wavelet_convs[i].weight is shape [4C, 1, K, K] for depthwise conv.
    return [ref_layer.wavelet_convs[i].weight.detach().contiguous() for i in range(levels)]


def _weights_for_opt(ref_layer, C, K, levels):
    # Convert [4C, 1, K, K] -> [C, 4, K, K]
    w = []
    for i in range(levels):
        w_orig = ref_layer.wavelet_convs[i].weight.detach().contiguous()
        w_fmt = w_orig.view(C, 4, K, K).contiguous()
        w.append(w_fmt)
    return w


def _call_forward(mod, mod_kind: str, x, weights, stride, pad, groups, dwt_scale, idwt_scale, training: bool):
    """
    Returns:
      out, saved_or_none
    """
    if mod_kind == "baseline":
        # baseline forward returns just out (no saved)
        out = mod.wtconv_forward(x, weights, stride, pad, groups, dwt_scale, idwt_scale)
        return out, None

    # optimized modules: return (out, saved) when called like in your tests
    out, saved = mod.wtconv_forward(
        x, weights,
        stride=stride, pad=pad, groups=groups,
        dwt_scale=dwt_scale, idwt_scale=idwt_scale,
        training=training,
    )
    return out, saved


def _call_forward_save_baseline(mod, x, weights, stride, pad, groups, dwt_scale, idwt_scale):
    # baseline save API
    return mod.wtconv_forward_save(x, weights, stride, pad, groups, dwt_scale, idwt_scale)


def _call_backward(mod, mod_kind: str, saved, grad_out, weights, groups, dwt_scale, idwt_scale):
    """
    Returns:
      grad_x, grad_w_list
    """
    if mod_kind == "baseline":
        # baseline backward signature in your tests: (saved, grad_out, weights, groups)
        return mod.wtconv_backward(saved, grad_out, weights, groups)

    # optimized signatures in your tests: (saved, grad_out, weights, groups, dwt_scale, idwt_scale)
    return mod.wtconv_backward(saved, grad_out, weights, groups, dwt_scale, idwt_scale)


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


# =============================================================================
# CORE equivalence tests (recommended target first)
# =============================================================================

@pytest.mark.parametrize("name,mod,kind", [
    pytest.param("cuda_module", cuda_module, "baseline",
                 marks=pytest.mark.skipif(cuda_module is None, reason="cuda_module not installed")),
    pytest.param("optimized_cuda_module", optimized_cuda_module, "opt",
                 marks=pytest.mark.skipif(optimized_cuda_module is None, reason="optimized_cuda_module not installed")),
    pytest.param("optimized2_cuda_module", optimized2_cuda_module, "opt",
                 marks=pytest.mark.skipif(optimized2_cuda_module is None, reason="optimized2_cuda_module not installed")),
])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_core_forward_backward_equivalence(name, mod, kind, cfg, wtconv_class):
    """
    Core correctness:
      - disables base branch in Reference
      - sets wavelet_scale=1 in Reference
      - compares:
          forward
          grad_input
          grad_weights (per level)
      - includes odd H/W to catch padding/crop bugs
    """
    B, C, H, W, K, levels = cfg
    device = torch.device("cuda")

    ref = _make_reference_model(wtconv_class, C=C, K=K, levels=levels, device=device, mode="core")

    x_ref = torch.randn(B, C, H, W, device=device, dtype=torch.float32, requires_grad=True)
    y_ref = ref(x_ref)

    # Random grad_out to avoid “all ones” masking issues
    grad_out = torch.randn_like(y_ref)
    grads = torch.autograd.grad(
        outputs=y_ref,
        inputs=[x_ref] + [ref.wavelet_convs[i].weight for i in range(levels)],
        grad_outputs=grad_out,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )
    grad_x_ref = grads[0].detach()
    grad_w_ref = [g.detach().contiguous() for g in grads[1:]]

    # Prepare inputs for CUDA path
    x = x_ref.detach().contiguous()

    pad = K // 2
    stride = 1
    # IMPORTANT: your modules appear to assume depthwise behavior internally.
    # Keep groups=1 exactly like your existing tests to avoid changing semantics here.
    groups = 1

    if kind == "baseline":
        weights = _weights_for_baseline(ref, levels)
        # baseline forward+save
        y_cuda, saved = _call_forward_save_baseline(mod, x, weights, stride, pad, groups, DWT_SCALE, IDWT_SCALE)
        gx_cuda, gw_cuda = _call_backward(mod, "baseline", saved, grad_out.detach().contiguous(), weights, groups, DWT_SCALE, IDWT_SCALE)

        # weight grads are in baseline format [4C,1,K,K]
        # grad_w_ref are also [4C,1,K,K] (same as Reference weights)
        gw_ref_cmp = grad_w_ref
        gw_cuda_cmp = gw_cuda

    else:
        # optimized weights format [C,4,K,K]
        weights = _weights_for_opt(ref, C, K, levels)

        # forward with training=True to get saved tensors
        y_cuda, saved = _call_forward(mod, "opt", x, weights, stride, pad, groups, DWT_SCALE, IDWT_SCALE, training=True)
        gx_cuda, gw_cuda = _call_backward(mod, "opt", saved, grad_out.detach().contiguous(), weights, groups, DWT_SCALE, IDWT_SCALE)

        # Reference grads need to be reshaped to [C,4,K,K] to compare
        gw_ref_cmp = [g.view(C, 4, K, K).contiguous() for g in grad_w_ref]
        gw_cuda_cmp = gw_cuda

    # Forward: shape must match
    assert y_ref.shape == y_cuda.shape, f"[{name}] forward shape mismatch: ref={tuple(y_ref.shape)} cuda={tuple(y_cuda.shape)}"
    fd = _max_abs_diff(y_ref.detach(), y_cuda.detach())
    assert fd < 1e-5, f"[{name}] forward max|diff|={fd:.3e} (cfg={cfg})"

    # Grad input
    assert grad_x_ref.shape == gx_cuda.shape, f"[{name}] grad_x shape mismatch"
    gxd = _max_abs_diff(grad_x_ref, gx_cuda)
    assert gxd < 1e-4, f"[{name}] grad_x max|diff|={gxd:.3e} (cfg={cfg})"

    # Grad weights
    assert len(gw_ref_cmp) == len(gw_cuda_cmp) == levels
    for i in range(levels):
        assert gw_ref_cmp[i].shape == gw_cuda_cmp[i].shape, f"[{name}] grad_w[{i}] shape mismatch"
        gwd = _max_abs_diff(gw_ref_cmp[i], gw_cuda_cmp[i])
        # weight grads are typically noisier; keep a slightly looser bound
        assert gwd < 1e-3, f"[{name}] grad_w[{i}] max|diff|={gwd:.3e} (cfg={cfg})"


# =============================================================================
# FULL equivalence placeholder (only valid if CUDA implements base branch + wavelet_scale)
# =============================================================================

@pytest.mark.skip(reason="Enable only after CUDA path implements base_conv/base_scale/wavelet_scale/do_stride to match full WTConv2d.")
def test_full_equivalence_placeholder():
    """
    When you're ready:
      - build Reference with mode="full"
      - run your CUDA layer that implements the full WTConv2d forward
      - compare end-to-end
    """
    pass
