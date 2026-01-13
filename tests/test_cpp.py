# import pytest
# import torch
# import numpy as np

# try: import cpp_module
# except ImportError: cpp_module = None

# @pytest.mark.skipif(cpp_module is None, reason="cpp_module not installed")
# def test_cpp_forward(config, reference_model):
#     B, C, H, W, K, levels = config
#     ref_layer = reference_model(C, K, levels, device='cpu')
#     x = torch.randn(B, C, H, W)
#     with torch.no_grad():
#         out_ref = ref_layer(x).numpy()
#     weights = [ref_layer.wavelet_convs[i].weight.detach() for i in range(levels)]
    
#     # Unified API call
#     out_cpp, _ = cpp_module.wtconv_forward(
#         x, weights, 1, K//2, 1, 0.5, 0.5, training=False
#     )
    
#     diff = np.abs(out_ref - out_cpp.numpy()).max()
#     assert diff < 1e-5

# @pytest.mark.skipif(cpp_module is None, reason="cpp_module not installed")
# def test_cpp_backward(config, reference_model):
#     B, C, H, W, K, levels = config
#     ref_layer = reference_model(C, K, levels, device='cpu')
#     x_ref = torch.randn(B, C, H, W, requires_grad=True)
#     out_ref = ref_layer(x_ref)
#     loss = out_ref.sum()
#     loss.backward()
#     grad_x_ref = x_ref.grad.detach()
#     grad_w_ref = [ref_layer.wavelet_convs[i].weight.grad.detach() for i in range(levels)]

#     x_cpp = x_ref.detach().clone()
#     weights_cpp = [ref_layer.wavelet_convs[i].weight.detach().clone() for i in range(levels)]

#     # Unified API: Training=True
#     out_cpp, saved_coeffs = cpp_module.wtconv_forward(
#         x_cpp, weights_cpp, 1, K//2, 1, 0.5, 0.5, training=True
#     )
    
#     grad_out = torch.ones_like(out_cpp)
#     grad_x_cpp, grad_w_cpp = cpp_module.wtconv_backward(
#         saved_coeffs, grad_out, weights_cpp, 1
#     )
    
#     diff_x = (grad_x_ref - grad_x_cpp).abs().max().item()
#     print(f"\n[dX] Max Diff: {diff_x:.8f}")
#     assert diff_x < 1e-4
    
#     for i in range(levels):
#         diff_w = (grad_w_ref[i] - grad_w_cpp[i]).abs().max().item()
#         print(f"[dW Level {i}] Max Diff: {diff_w:.8f}")
#         assert grad_w_ref[i].shape == grad_w_cpp[i].shape
#         assert diff_w < 1e-4