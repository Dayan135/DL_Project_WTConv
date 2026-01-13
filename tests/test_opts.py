# tests/test_ops.py

import pytest
import torch
import numpy as np
import sys
import os

# --- Path Setup ---
# Get the root directory (parent of 'tests')
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the source folders to sys.path so Python can find the .pyd files
sys.path.append(os.path.join(HERE, "cuda_source"))
sys.path.append(os.path.join(HERE, "optimized_cuda_source"))
sys.path.append(os.path.join(HERE, "optimized2_cuda_source"))

# --- Import Modules ---
try: import cuda_module
except ImportError: cuda_module = None

try: import optimized_cuda_module
except ImportError: optimized_cuda_module = None

try: import optimized2_cuda_module
except ImportError: optimized2_cuda_module = None

DWT_SCALE = 0.5
IDWT_SCALE = 0.5


# =============================================================================
# 1. Baseline Tests (cuda_module)
# =============================================================================

@pytest.mark.skipif(cuda_module is None, reason="cuda_module not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_baseline_cuda_forward(config, reference_model):
    """Checks that Baseline Forward matches Reference output."""
    B, C, H, W, K, levels = config
    device = torch.device('cuda')
    
    ref_layer = reference_model(C, K, levels, device=device)
    x = torch.randn(B, C, H, W, device=device)
    
    with torch.no_grad():
        out_ref = ref_layer(x).cpu().numpy()
        
    weights = [ref_layer.wavelet_convs[i].weight.detach() for i in range(levels)]
    out_cuda = cuda_module.wtconv_forward(
        x, weights, 1, K//2, 1, DWT_SCALE, IDWT_SCALE
    )
    
    diff = np.abs(out_ref - out_cuda.cpu().numpy()).max()
    assert diff < 1e-5, f"Max Diff: {diff:.8f}"


@pytest.mark.skipif(cuda_module is None, reason="cuda_module not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_baseline_cuda_backward(config, reference_model):
    """Checks that Baseline Backward gradients match Reference Autograd."""
    B, C, H, W, K, levels = config
    device = torch.device('cuda')
    
    ref_layer = reference_model(C, K, levels, device=device)
    x_ref = torch.randn(B, C, H, W, device=device, requires_grad=True)
    out_ref = ref_layer(x_ref)
    loss = out_ref.sum()
    loss.backward()
    
    grad_x_ref = x_ref.grad.detach()
    grad_w_ref = [ref_layer.wavelet_convs[i].weight.grad.detach() for i in range(levels)]

    x_cuda = x_ref.detach().clone()
    weights_cuda = [ref_layer.wavelet_convs[i].weight.detach().clone() for i in range(levels)]
    
    out_cuda, saved_coeffs = cuda_module.wtconv_forward_save(
        x_cuda, weights_cuda, 1, K//2, 1, DWT_SCALE, IDWT_SCALE
    )
    grad_out = torch.ones_like(out_cuda)
    grad_x_cuda, grad_w_cuda = cuda_module.wtconv_backward(
        saved_coeffs, grad_out, weights_cuda, 1
    )
    
    diff_x = (grad_x_ref - grad_x_cuda).abs().max().item()
    assert diff_x < 1e-3, f"Input Gradient mismatch! Max diff: {diff_x}"
    
    for i in range(levels):
        diff_w = (grad_w_ref[i] - grad_w_cuda[i]).abs().max().item()
        assert diff_w < 1e-3, f"Weight Gradient mismatch at level {i}! Max diff: {diff_w}"


# =============================================================================
# 2. Optimized V1 Tests (optimized_cuda_module)
# =============================================================================

@pytest.mark.skipif(optimized_cuda_module is None, reason="optimized_cuda_module not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_optimized_forward(config, reference_model):
    B, C, H, W, K, levels = config
    device = torch.device("cuda")

    ref_layer = reference_model(C, K, levels, device=device)
    x = torch.randn(B, C, H, W, device=device)

    with torch.no_grad():
        out_ref = ref_layer(x).cpu().numpy()

    weights_opt = []
    for i in range(levels):
        w_original = ref_layer.wavelet_convs[i].weight.detach()
        w_formatted = w_original.view(C, 4, K, K).contiguous()
        weights_opt.append(w_formatted)

    res = optimized_cuda_module.wtconv_forward(
        x, weights_opt, stride=1, pad=K // 2, groups=1, dwt_scale=DWT_SCALE, idwt_scale=IDWT_SCALE, training=False,
    )
    out_opt = res[0]

    diff = np.abs(out_ref - out_opt.detach().cpu().numpy()).max()
    assert diff < 1e-5, f"Max Diff: {diff:.8f}"


@pytest.mark.skipif(optimized_cuda_module is None, reason="optimized_cuda_module not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_optimized_backward_matches_reference(config, reference_model):
    """Verifies V1 Backward gradients match Reference."""
    B, C, H, W, K, levels = config
    device = torch.device("cuda")

    ref_layer = reference_model(C, K, levels, device=device).to(device)
    x_ref = torch.randn(B, C, H, W, device=device, requires_grad=True)

    weights_opt = []
    ref_weight_params = []
    for i in range(levels):
        w_original = ref_layer.wavelet_convs[i].weight
        ref_weight_params.append(w_original)
        w_formatted = w_original.detach().view(C, 4, K, K).contiguous()
        weights_opt.append(w_formatted)

    out_ref = ref_layer(x_ref)
    grad_out = torch.randn_like(out_ref)

    grads = torch.autograd.grad(
        outputs=out_ref, inputs=[x_ref] + ref_weight_params, grad_outputs=grad_out,
        retain_graph=False, create_graph=False, allow_unused=False,
    )

    grad_x_ref = grads[0].detach()
    grad_w_ref_fmt = [g.detach().view(C, 4, K, K).contiguous() for g in grads[1:]]

    x_opt = x_ref.detach()
    res = optimized_cuda_module.wtconv_forward(
        x_opt, weights_opt, 1, K // 2, 1, DWT_SCALE, IDWT_SCALE, True,
    )
    saved = res[1]

    bw = optimized_cuda_module.wtconv_backward(
        saved, grad_out, weights_opt, 1, DWT_SCALE, IDWT_SCALE,
    )
    grad_x_opt, grad_w_opt = bw[0], bw[1]

    max_diff_x = (grad_x_ref - grad_x_opt).abs().max().item()
    assert max_diff_x < 1e-4, f"grad_input max diff: {max_diff_x:.8e}"

    for i in range(levels):
        max_diff_w = (grad_w_ref_fmt[i] - grad_w_opt[i]).abs().max().item()
        assert max_diff_w < 1e-3, f"grad_weight[{i}] max diff: {max_diff_w:.8e}"


# =============================================================================
# 3. Optimized V2 Tests (optimized2_cuda_module - Fused Split)
# =============================================================================

@pytest.mark.skipif(optimized2_cuda_module is None, reason="optimized2_cuda_module not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_optimized2_forward(config, reference_model):
    """Verifies Forward pass of V2 (should match Reference)"""
    B, C, H, W, K, levels = config
    device = torch.device("cuda")

    ref_layer = reference_model(C, K, levels, device=device)
    x = torch.randn(B, C, H, W, device=device)

    with torch.no_grad():
        out_ref = ref_layer(x).cpu().numpy()

    weights_opt = []
    for i in range(levels):
        w_original = ref_layer.wavelet_convs[i].weight.detach()
        w_formatted = w_original.view(C, 4, K, K).contiguous()
        weights_opt.append(w_formatted)

    # Call Optimized V2
    res = optimized2_cuda_module.wtconv_forward(
        x, weights_opt, stride=1, pad=K // 2, groups=1, dwt_scale=DWT_SCALE, idwt_scale=IDWT_SCALE, training=False,
    )
    out_opt = res[0]

    diff = np.abs(out_ref - out_opt.detach().cpu().numpy()).max()
    assert diff < 1e-5, f"V2 Forward Max Diff: {diff:.8f}"


@pytest.mark.skipif(optimized2_cuda_module is None, reason="optimized2_cuda_module not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_optimized2_backward_matches_reference(config, reference_model):
    """
    Verifies Backward pass of V2 (Fused Split).
    """
    B, C, H, W, K, levels = config
    device = torch.device("cuda")

    ref_layer = reference_model(C, K, levels, device=device).to(device)
    x_ref = torch.randn(B, C, H, W, device=device, requires_grad=True)

    weights_opt = []
    ref_weight_params = []
    for i in range(levels):
        w_original = ref_layer.wavelet_convs[i].weight
        ref_weight_params.append(w_original)
        w_formatted = w_original.detach().view(C, 4, K, K).contiguous()
        weights_opt.append(w_formatted)

    # 1. Reference Autograd
    out_ref = ref_layer(x_ref)
    grad_out = torch.randn_like(out_ref)

    grads = torch.autograd.grad(
        outputs=out_ref, inputs=[x_ref] + ref_weight_params, grad_outputs=grad_out,
        retain_graph=False, create_graph=False, allow_unused=False,
    )
    grad_x_ref = grads[0].detach()
    grad_w_ref_fmt = [g.detach().view(C, 4, K, K).contiguous() for g in grads[1:]]

    # 2. Optimized V2
    x_opt = x_ref.detach()
    res = optimized2_cuda_module.wtconv_forward(
        x_opt, weights_opt, 1, K // 2, 1, DWT_SCALE, IDWT_SCALE, True,
    )
    saved = res[1]

    bw = optimized2_cuda_module.wtconv_backward(
        saved, grad_out, weights_opt, 1, DWT_SCALE, IDWT_SCALE,
    )
    grad_x_opt, grad_w_opt = bw[0], bw[1]

    # 3. Compare
    max_diff_x = (grad_x_ref - grad_x_opt).abs().max().item()
    assert max_diff_x < 1e-4, f"V2 grad_input max diff: {max_diff_x:.8e}"

    for i in range(levels):
        max_diff_w = (grad_w_ref_fmt[i] - grad_w_opt[i]).abs().max().item()
        assert max_diff_w < 1e-4, f"V2 grad_weight[{i}] max diff: {max_diff_w:.8e}"