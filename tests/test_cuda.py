import pytest
import torch
import numpy as np

try: import cuda_module
except ImportError: cuda_module = None

@pytest.mark.skipif(cuda_module is None, reason="cuda_module not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_baseline_cuda_forward(config, reference_model):
    B, C, H, W, K, levels = config
    device = torch.device('cuda')
    
    # 1. Setup Reference (CUDA)
    ref_layer = reference_model(C, K, levels, device=device)
    x = torch.randn(B, C, H, W, device=device)
    
    # 2. Run Reference
    with torch.no_grad():
        out_ref = ref_layer(x).cpu().numpy()
        
    # 3. Run Baseline Module
    # CORRECTION: Pass a Python list of tensors, do NOT wrap in torch.tensor()
    weights = [ref_layer.wavelet_convs[i].weight.detach() for i in range(levels)]
    out_cuda = cuda_module.wtconv_forward(
        x, weights, 1, K//2, 1, 0.5, 0.5
    )
    
    # 4. Verify
    diff = np.abs(out_ref - out_cuda.cpu().numpy()).max()
    assert diff < 1e-5, f"Max Diff: {diff:.8f}"


@pytest.mark.skipif(cuda_module is None, reason="cuda_module not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_baseline_cuda_backward(config, reference_model):
    B, C, H, W, K, levels = config
    device = torch.device('cuda')
    
    # --- 1. Run Reference (PyTorch Autograd) ---
    # Create model with bias disabled (standardized fixture)
    ref_layer = reference_model(C, K, levels, device=device)
    
    # Input requires grad to check dX
    x_ref = torch.randn(B, C, H, W, device=device, requires_grad=True)
    
    # Forward
    out_ref = ref_layer(x_ref)
    
    # Backward (Scalar loss sum)
    # This effectively sets grad_output to all 1.0
    loss = out_ref.sum()
    loss.backward()
    
    # Capture "Truth" Gradients
    grad_x_ref = x_ref.grad.detach()
    grad_w_ref = [ref_layer.wavelet_convs[i].weight.grad.detach() for i in range(levels)]

    # --- 2. Run Baseline CUDA (Manual Backward) ---
    # Use detached clones so we don't engage Autograd
    x_cuda = x_ref.detach().clone()
    weights_cuda = [ref_layer.wavelet_convs[i].weight.detach().clone() for i in range(levels)]
    
    # A. Forward (Save Tensors)
    # Returns: (output, saved_coeffs_list)
    out_cuda, saved_coeffs = cuda_module.wtconv_forward_save(
        x_cuda, weights_cuda, 1, K//2, 1, 0.5, 0.5
    )
    
    # B. Backward
    # Create grad_output matching the Reference loss (all ones)
    grad_out = torch.ones_like(out_cuda)
    
    # Returns: (grad_input, grad_weights_list)
    grad_x_cuda, grad_w_cuda = cuda_module.wtconv_backward(
        saved_coeffs, grad_out, weights_cuda, 1
    )
    
    # --- 3. Verify ---
    
    # A. Check Input Gradients (dX)
    diff_x = (grad_x_ref - grad_x_cuda).abs().max().item()
    print(f"\n[dX] Max Diff: {diff_x:.8f}")
    assert diff_x < 1e-3, f"Input Gradient mismatch! Max diff: {diff_x}"
    
    # B. Check Weight Gradients (dW)
    for i in range(levels):
        ref_gw = grad_w_ref[i]
        cuda_gw = grad_w_cuda[i]
        
        diff_w = (ref_gw - cuda_gw).abs().max().item()
        print(f"[dW Level {i}] Max Diff: {diff_w:.8f}")
        
        assert ref_gw.shape == cuda_gw.shape, f"Shape mismatch at level {i}"
        assert diff_w < 1e-3, f"Weight Gradient mismatch at level {i}! Max diff: {diff_w}"