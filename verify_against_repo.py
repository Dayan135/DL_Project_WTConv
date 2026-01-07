import sys
import os
import torch
import numpy as np

# 1. Setup Paths
# Add 'Reference' to path so we can import the original BGU code
repo_path = os.path.join(os.path.dirname(__file__), 'Reference')
sys.path.append(repo_path)

# Add 'cpp_source' to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'cpp_source'))

# 2. Import Modules
try:
    import cpp_module
    print("[INFO] Loaded compiled C++ module.")
except ImportError:
    print("[ERROR] Could not load cpp_module. Run 'python setup.py build_ext --inplace' in cpp_source/ first.")
    sys.exit(1)

try:
    # Try importing based on standard folder structure
    from wtconv.wtconv2d import WTConv2d
    print("[INFO] Loaded WTConv2d from Reference repository.")
except ImportError:
    try:
        # Fallback if __init__ handles exports
        from wtconv import WTConv2d
        print("[INFO] Loaded WTConv2d from Reference repository (direct package).")
    except ImportError:
        print("[ERROR] Could not import WTConv2d. Check if 'Reference/wtconv' exists.")
        sys.exit(1)

def verify_repo_equivalence():
    print("\n=== Verifying C++ Kernel against Actual BGU Repo Implementation ===")

    # --- Configuration ---
    # We use wt_levels=1 because your C++ kernel is currently hardcoded for 1 level.
    # The BGU repo supports multiple, but we verify the single-level logic first.
    BATCH = 2
    CHANNELS = 4
    HEIGHT = 32
    WIDTH = 32
    KERNEL_SIZE = 5
    
    # 1. Instantiate the Reference Layer
    # Note: WTConv2d(in_channels, out_channels, kernel_size=5, wt_levels=1)
    ref_layer = WTConv2d(CHANNELS, CHANNELS, kernel_size=KERNEL_SIZE, wt_levels=1)
    ref_layer.eval() # Set to eval mode (disable dropout etc if present)

    # 2. Create Inputs
    x_torch = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
    
    # 3. Extract Weights & Config from Reference
    # The BGU implementation stores the conv layer in 'self.wt_conv'
    if not hasattr(ref_layer, 'wt_conv'):
        print("[ERROR] The imported WTConv2d does not have 'wt_conv'. Repo structure might differ.")
        return

    weight_torch = ref_layer.wt_conv.weight.detach()
    
    # DETECT GROUPS DYNAMICALLY
    # If shape is (Out, 1, K, K) -> Depthwise (Groups = Out)
    # If shape is (Out, In, K, K) -> Dense (Groups = 1)
    out_channels_wt, in_channels_wt, k, _ = weight_torch.shape
    
    if in_channels_wt == 1:
        groups = out_channels_wt
        print(f"[INFO] Detected Depthwise Convolution (Groups={groups})")
    else:
        groups = 1
        print(f"[INFO] Detected Dense Convolution (Groups=1)")

    # 4. Run Reference Forward
    print("Running Reference Forward...")
    with torch.no_grad():
        out_ref = ref_layer(x_torch).numpy()

    # 5. Run C++ Forward
    print("Running C++ Forward...")
    x_np = x_torch.numpy()
    w_np = weight_torch.numpy()
    
    # WTConv usually sets padding = kernel_size // 2
    stride = 1
    pad = KERNEL_SIZE // 2
    
    out_cpp = cpp_module.wtconv_forward(x_np, w_np, stride, pad, groups)

    # 6. Compare
    print("\n--- Comparison Results ---")
    print(f"Reference Output Shape: {out_ref.shape}")
    print(f"C++ Output Shape:       {out_cpp.shape}")
    
    diff = np.abs(out_ref - out_cpp).max()
    print(f"Max Absolute Difference: {diff:.8f}")

    if diff < 1e-5:
        print("\n✅ MATCH CONFIRMED: Your C++ code is equivalent to the BGU Repo!")
    else:
        print("\n❌ MISMATCH detected.")

if __name__ == "__main__":
    verify_repo_equivalence()