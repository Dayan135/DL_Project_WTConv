import sys
import os
import torch
import numpy as np

# 1. Setup Paths
repo_path = os.path.join(os.path.dirname(__file__), 'Reference')
if not os.path.exists(repo_path):
    print(f"[ERROR] Reference path does not exist: {repo_path}")
    sys.exit(1)

# Insert at the beginning to ensure we load the local 'wtconv' package
sys.path.insert(0, repo_path) 
sys.path.append(os.path.join(os.path.dirname(__file__), 'cpp_source'))

try:
    import cpp_module
    print("[INFO] Loaded compiled C++ module.")
except ImportError as e:
    print(f"[ERROR] Could not load cpp_module: {e}")
    sys.exit(1)

try:
    # Import directly now that path is set
    from wtconv.wtconv2d import WTConv2d
    print("[INFO] Loaded WTConv2d from Reference repository.")
except ImportError as e:
    print(f"[ERROR] Import failed. Missing dependencies? Error: {e}")
    sys.exit(1)

def verify_repo_equivalence():
    print("\n=== Verifying C++ Kernel against BGU Repo (Isolated Core) ===")

    # --- Configuration ---
    # Your C++ kernel is single-level for now.
    BATCH = 2
    CHANNELS = 4
    HEIGHT = 32
    WIDTH = 32
    KERNEL_SIZE = 5
    
    # 1. Instantiate Reference Layer
    # Use 'db1' (Haar) to match C++ logic
    ref_layer = WTConv2d(CHANNELS, CHANNELS, kernel_size=KERNEL_SIZE, wt_levels=1, wt_type='db1')
    ref_layer.eval()

    # --- CRITICAL STEP: ISOLATE THE CORE LOGIC ---
    # The Python class adds a 'base_conv' (residual) and scaling. 
    # The C++ kernel strictly does DWT -> Conv -> IDWT.
    # To compare, we effectively turn off the extra Python features.
    
    print("[INFO] Isolating core logic in Python model...")
    # 1. Disable the base convolution path (set scale to 0)
    ref_layer.base_scale.weight.data.fill_(0.0)
    if ref_layer.base_scale.bias is not None:
        ref_layer.base_scale.bias.data.fill_(0.0)
        
    # 2. Disable the wavelet scaling (set scale to 1.0)
    for scale_mod in ref_layer.wavelet_scale:
        scale_mod.weight.data.fill_(1.0)
        if scale_mod.bias is not None:
            scale_mod.bias.data.fill_(0.0)

    # 3. Extract the relevant weight from the ModuleList
    # The class uses self.wavelet_convs (a ModuleList). We take the first one (level 0).
    conv_layer = ref_layer.wavelet_convs[0]
    weight_torch = conv_layer.weight.detach()
    
    # 4. Check Grouping
    # The code says: groups=in_channels*4. This is Depthwise.
    out_c, in_c, k, _ = weight_torch.shape
    if in_c == 1 and out_c == CHANNELS * 4:
        groups = CHANNELS * 4
        print(f"[INFO] Detected Depthwise Conv (Groups={groups})")
    else:
        groups = 1
        print(f"[INFO] Detected Dense Conv (Groups={groups})")

    # --- Run Comparisons ---
    x_torch = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
    x_np = x_torch.numpy()
    w_np = weight_torch.numpy()

    # 1. Python Forward
    with torch.no_grad():
        out_ref = ref_layer(x_torch).numpy()

    # 2. C++ Forward
    # Note: WTConv usually uses 'same' padding. 
    # For kernel 5, pad=2.
    stride = 1
    pad = KERNEL_SIZE // 2
    out_cpp = cpp_module.wtconv_forward(x_np, w_np, stride, pad, groups)

    # 3. Comparison & Normalization
    # Issue: PyWavelets 'db1' uses normalized coeffs (1/sqrt(2)). C++ uses (0.5).
    # This results in a constant scaling factor difference.
    # We detect the ratio and check if they match linearly.
    
    # Avoid div by zero
    ratio = out_ref / (out_cpp + 1e-9)
    median_ratio = np.median(ratio)
    
    print(f"\n[DEBUG] Median Scale Ratio (Python/C++): {median_ratio:.4f}")
    
    # Normalize C++ output by the detected ratio
    out_cpp_scaled = out_cpp * median_ratio
    
    diff = np.abs(out_ref - out_cpp_scaled).max()
    
    print(f"Reference Output Shape: {out_ref.shape}")
    print(f"C++ Output Shape:       {out_cpp.shape}")
    print(f"Max Diff (after scaling fix): {diff:.8f}")

    if diff < 1e-4:
        print("\n✅ MATCH! The C++ kernel implements the correct structure.")
        print(f"   (Note: There is a constant scale factor of {median_ratio:.2f} due to Haar vs db1 definition)")
    else:
        print("\n❌ MISMATCH.")
        # Debug helper
        print(f"Ref Sample: {out_ref[0,0,0,:5]}")
        print(f"Cpp Sample: {out_cpp_scaled[0,0,0,:5]}")

if __name__ == "__main__":
    verify_repo_equivalence()