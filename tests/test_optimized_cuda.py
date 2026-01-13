import pytest
import torch
import numpy as np

try:
    import optimized_cuda_module
except ImportError:
    optimized_cuda_module = None

DWT_SCALE = 0.5
IDWT_SCALE = 0.5


@pytest.mark.skipif(optimized_cuda_module is None, reason="optimized_cuda_module not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_optimized_forward(config, reference_model):
    B, C, H, W, K, levels = config
    device = torch.device("cuda")

    # 1. Setup Reference
    ref_layer = reference_model(C, K, levels, device=device)
    x = torch.randn(B, C, H, W, device=device)

    # 2. Run Reference
    with torch.no_grad():
        out_ref = ref_layer(x).cpu().numpy()

    # 3. Prepare Inputs for Optimized Module
    weights_opt = []
    for i in range(levels):
        w_original = ref_layer.wavelet_convs[i].weight.detach()  # (4C,1,K,K)
        w_formatted = w_original.view(C, 4, K, K).contiguous()
        weights_opt.append(w_formatted)

    # 4. Run Optimized Module (avoid tuple unpacking)
    res = optimized_cuda_module.wtconv_forward(
        x,
        weights_opt,
        stride=1,
        pad=K // 2,
        groups=1,
        dwt_scale=DWT_SCALE,
        idwt_scale=IDWT_SCALE,
        training=False,
    )
    out_opt = res[0]

    # 5. Verify
    diff = np.abs(out_ref - out_opt.detach().cpu().numpy()).max()
    assert diff < 1e-5, f"Max Diff: {diff:.8f}"


@pytest.mark.skipif(optimized_cuda_module is None, reason="optimized_cuda_module not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_optimized_forward_training_mode(config, reference_model):
    """Verifies that training mode correctly returns saved tensors."""
    B, C, H, W, K, levels = config
    device = torch.device("cuda")

    x = torch.randn(B, C, H, W, device=device)
    weights = [torch.randn(C, 4, K, K, device=device) for _ in range(levels)]

    res = optimized_cuda_module.wtconv_forward(
        x,
        weights,
        1,
        K // 2,
        1,
        DWT_SCALE,
        IDWT_SCALE,
        True,
    )
    out = res[0]
    saved = res[1]

    assert len(saved) == levels
    assert saved[0].shape == x.shape
    assert out is not None


@pytest.mark.skipif(optimized_cuda_module is None, reason="optimized_cuda_module not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_optimized_backward_matches_reference(config, reference_model):
    """
    Compares:
      - grad_input from optimized wtconv_backward
      - grad_weights from optimized wtconv_backward
    against PyTorch autograd grads from the reference model, using the SAME weights.
    """
    B, C, H, W, K, levels = config
    device = torch.device("cuda")

    ref_layer = reference_model(C, K, levels, device=device).to(device)

    x_ref = torch.randn(B, C, H, W, device=device, requires_grad=True)

    weights_opt = []
    ref_weight_params = []
    for i in range(levels):
        w_original = ref_layer.wavelet_convs[i].weight  # (4C,1,K,K)
        ref_weight_params.append(w_original)
        w_formatted = w_original.detach().view(C, 4, K, K).contiguous()
        weights_opt.append(w_formatted)

    # Reference backward
    out_ref = ref_layer(x_ref)
    grad_out = torch.randn_like(out_ref)

    grads = torch.autograd.grad(
        outputs=out_ref,
        inputs=[x_ref] + ref_weight_params,
        grad_outputs=grad_out,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )

    grad_x_ref = grads[0].detach()
    grad_w_ref_list = [g.detach() for g in grads[1:]]

    grad_w_ref_fmt = []
    for i in range(levels):
        gw = grad_w_ref_list[i]  # (4C,1,K,K)
        gw_fmt = gw.view(C, 4, K, K).contiguous()
        grad_w_ref_fmt.append(gw_fmt)

    # Optimized forward (training=True) + optimized backward
    x_opt = x_ref.detach()

    res = optimized_cuda_module.wtconv_forward(
        x_opt,
        weights_opt,
        1,
        K // 2,
        1,
        DWT_SCALE,
        IDWT_SCALE,
        True,
    )
    out_opt = res[0]
    saved = res[1]

    # UPDATED CALL: backward now takes dwt_scale + idwt_scale
    bw = optimized_cuda_module.wtconv_backward(
        saved,
        grad_out,
        weights_opt,
        1,
        DWT_SCALE,
        IDWT_SCALE,
    )
    grad_x_opt = bw[0]
    grad_w_opt = bw[1]

    assert isinstance(grad_w_opt, (list, tuple))
    assert len(grad_w_opt) == levels
    for i in range(levels):
        assert grad_w_opt[i].shape == weights_opt[i].shape

    max_diff_x = (grad_x_ref - grad_x_opt).abs().max().item()
    assert max_diff_x < 1e-4, f"grad_input max diff: {max_diff_x:.8e}"

    for i in range(levels):
        max_diff_w = (grad_w_ref_fmt[i] - grad_w_opt[i]).abs().max().item()
        assert max_diff_w < 1e-4, f"grad_weight[{i}] max diff: {max_diff_w:.8e}"


@pytest.mark.skipif(optimized_cuda_module is None, reason="optimized_cuda_module not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_optimized_backward_shapes_and_finiteness(config):
    """
    Pure contract test:
      - wtconv_forward(training=True) returns saved tensors
      - wtconv_backward returns correct shapes
      - outputs are finite
    """
    B, C, H, W, K, levels = config
    device = torch.device("cuda")

    x = torch.randn(B, C, H, W, device=device)
    weights = [torch.randn(C, 4, K, K, device=device) for _ in range(levels)]

    res = optimized_cuda_module.wtconv_forward(
        x,
        weights,
        1,
        K // 2,
        1,
        DWT_SCALE,
        IDWT_SCALE,
        True,
    )
    out = res[0]
    saved = res[1]

    assert out.shape == (B, C, H, W)
    assert isinstance(saved, (list, tuple))
    assert len(saved) == levels
    assert saved[0].shape == x.shape

    grad_out = torch.randn_like(out)

    bw = optimized_cuda_module.wtconv_backward(
        saved,
        grad_out,
        weights,
        1,
        DWT_SCALE,
        IDWT_SCALE,
    )
    grad_x = bw[0]
    grad_w = bw[1]

    assert grad_x.shape == x.shape
    assert len(grad_w) == levels
    for i in range(levels):
        assert grad_w[i].shape == weights[i].shape

    assert torch.isfinite(out).all()
    assert torch.isfinite(grad_x).all()
    for i in range(levels):
        assert torch.isfinite(grad_w[i]).all()
