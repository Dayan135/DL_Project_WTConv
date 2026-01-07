import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# Import your compiled C++ module
sys.path.append(os.path.join(os.path.dirname(__file__), 'cpp_source'))
try:
    import cpp_module
except ImportError:
    print("Error: Could not import cpp_module. Did you compile it?")
    sys.exit(1)

def pytorch_haar_dwt(x):
    """ PyTorch implementation of the exact 2D Haar DWT logic. """
    N, C, H, W = x.shape
    
    # 1. Extract 2x2 patches
    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]
    
    # 2. Haar Transform (matches your C++ 0.5 scaling)
    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 - x01 + x10 - x11) * 0.5
    hl = (x00 + x01 - x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    
    # 3. Stack carefully to match C++ Interleaved layout:
    out = torch.stack([ll, lh, hl, hh], dim=2) 
    out = out.view(N, C * 4, H // 2, W // 2)   
    return out

def pytorch_haar_idwt(x):
    """ PyTorch implementation of Inverse Haar DWT. """
    N, C4, H_sub, W_sub = x.shape
    C = C4 // 4
    
    x_reshaped = x.view(N, C, 4, H_sub, W_sub)
    ll = x_reshaped[:, :, 0, :, :]
    lh = x_reshaped[:, :, 1, :, :]
    hl = x_reshaped[:, :, 2, :, :]
    hh = x_reshaped[:, :, 3, :, :]
    
    x00 = (ll + lh + hl + hh) * 0.5
    x01 = (ll - lh + hl - hh) * 0.5
    x10 = (ll + lh - hl - hh) * 0.5
    x11 = (ll - lh - hl + hh) * 0.5
    
    out = torch.zeros(N, C, H_sub * 2, W_sub * 2, dtype=x.dtype, device=x.device)
    out[:, :, 0::2, 0::2] = x00
    out[:, :, 0::2, 1::2] = x01
    out[:, :, 1::2, 0::2] = x10
    out[:, :, 1::2, 1::2] = x11
    return out

def verify_mode(mode_name, groups_config):
    print(f"\n=== Testing Mode: {mode_name} ===")
    
    # --- Config ---
    N, C, H, W = 2, 2, 8, 8 
    K = 3
    Stride = 1
    Pad = 1
    
    # Wavelet domain channels
    C_wt = 4 * C
    
    # Determine groups based on config
    if groups_config == "dense":
        groups = 1
    elif groups_config == "depthwise":
        groups = C_wt # Depthwise in wavelet domain
    else:
        groups = groups_config

    print(f"Config: Input=({N},{C},{H},{W}), Groups={groups}")

    # --- 1. Inputs ---
    x_torch = torch.randn(N, C, H, W, dtype=torch.float32)
    
    # Weight Shape Logic:
    # PyTorch weights are (Out, In/Groups, K, K)
    # If dense: (4*C, 4*C, K, K)
    # If depthwise: (4*C, 1, K, K)
    w_torch = torch.randn(C_wt, C_wt // groups, K, K, dtype=torch.float32)
    
    x_np = x_torch.numpy()
    w_np = w_torch.numpy()

    # --- 2. PyTorch Reference ---
    # A. DWT
    wt_out = pytorch_haar_dwt(x_torch)
    
    # B. Conv2d (with Groups!)
    conv_out = F.conv2d(wt_out, w_torch, stride=Stride, padding=Pad, groups=groups)
    
    # C. IDWT
    final_out_torch = pytorch_haar_idwt(conv_out)

    # --- 3. C++ Implementation ---
    # Now passing 'groups' argument
    final_out_cpp = cpp_module.wtconv_forward(x_np, w_np, Stride, Pad, groups)

    # --- 4. Compare ---
    diff = np.abs(final_out_torch.numpy() - final_out_cpp).max()
    print(f"Max Absolute Difference: {diff:.8f}")

    if diff < 1e-5:
        print(f"✅ {mode_name}: SUCCESS")
    else:
        print(f"❌ {mode_name}: FAILURE")

if __name__ == "__main__":
    # Test 1: Old behavior (Dense)
    verify_mode("Dense Conv (Groups=1)", "dense")
    
    # Test 2: New behavior (Depthwise / Grouped) -> This mimics WTConv Repo
    verify_mode("Depthwise Conv (Groups=4*C)", "depthwise")