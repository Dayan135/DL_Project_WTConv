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
sys.path.append(os.path.join(os.path.dirname(__file__), 'cuda_source'))

# 2. Import Modules
try:
    import cpp_module
    print("[INFO] Loaded compiled C++ module.")
except ImportError as e:
    print(f"[ERROR] Could not load cpp_module: {e}")

try:
    import cuda_module
    print("[INFO] Loaded compiled CUDA module.")
except ImportError as e:
    print(f"[ERROR] Could not load cuda_module: {e}")

try:
    # Import directly now that path is set
    from wtconv.wtconv2d import WTConv2d
    print("[INFO] Loaded WTConv2d from Reference repository.")
except ImportError as e:
    print(f"[ERROR] Import failed. Missing dependencies? Error: {e}")
    sys.exit(1)

def verify_repo_equivalence():
    print("\n=== Verifying Kernels against BGU Repo (Isolated Core) ===")

    # --- Configuration ---
    BATCH = 2
    CHANNELS = 4
    HEIGHT = 32
    WIDTH = 32
    KERNEL_SIZE = 5
    
    # 1. Instantiate Reference Layer
    # Use 'db1' (Haar) to match our Kernel logic
    ref_layer = WTConv2d(CHANNELS, CHANNELS, kernel_size=KERNEL_SIZE, wt_levels=1, wt_type='db1')
    ref_layer.eval()

    # --- ISOLATE CORE LOGIC ---
    print("[INFO] Isolating core logic in Python model...")
    ref_layer.base_scale.weight.data.fill_(0.0)
    if ref_layer.base_scale.bias is not None:
        ref_layer.base_scale.bias.data.fill_(0.0)
        
    for scale_mod in ref_layer.wavelet_scale:
        scale_mod.weight.data.fill_(1.0)
        if scale_mod.bias is not None:
            scale_mod.bias.data.fill_(0.0)

    # Extract Weight
    conv_layer = ref_layer.wavelet_convs[0]
    weight_torch = conv_layer.weight.detach()
    
    # Check Grouping
    out_c, in_c, k, _ = weight_torch.shape
    if in_c == 1 and out_c == CHANNELS * 4:
        groups = CHANNELS * 4
        print(f"[INFO] Detected Depthwise Conv (Groups={groups})")
    else:
        groups = 1
        print(f"[INFO] Detected Dense Conv (Groups={groups})")

    # --- Prepare Inputs ---
    x_torch = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
    x_np = x_torch.numpy()
    w_np = weight_torch.numpy()
    
    stride = 1
    pad = KERNEL_SIZE // 2

    # ====================================================
    # 1. Run Reference (Python)
    # ====================================================
    print("\n--- 1. Running Reference (Python/CPU) ---")
    with torch.no_grad():
        out_ref = ref_layer(x_torch).numpy()

    # ====================================================
    # 2. Run C++ Kernel
    # ====================================================
    if 'cpp_module' in sys.modules:
        print("\n--- 2. Running C++ Kernel ---")
        # We pass scales 0.5, 0.5 to match 'db1' normalization
        out_cpp = cpp_module.wtconv_forward(x_np, w_np, stride, pad, groups, 0.5, 0.5)
        
        diff_cpp = np.abs(out_ref - out_cpp).max()
        print(f"C++ Max Diff: {diff_cpp:.8f}")

        if diff_cpp < 1e-5:
            print("✅ C++ MATCH")
        else:
            print("❌ C++ MISMATCH")
    
    # ====================================================
    # 3. Run CUDA Kernel
    # ====================================================
    if 'cuda_module' in sys.modules and torch.cuda.is_available():
        print("\n--- 3. Running CUDA Kernel ---")
        
        # Move inputs to GPU
        x_cuda = x_torch.cuda()
        w_cuda = weight_torch.cuda()
        
        # Call CUDA Kernel
        # Note: We use the atomic forward wrapper exposed in cuda_module
        # Pass 0.5 scales to match db1
        out_cuda_tensor = cuda_module.conv_forward(
            cuda_module.dwt_forward(x_cuda, 0.5), # DWT
            w_cuda, 
            stride, 
            pad
        )
        
        # We need to manually IDWT because our cuda_module exposes atomic ops now
        # OR if you implemented the 'wtconv_forward' composite in pybind_module, call that.
        # Assuming we exposed 'wtconv_forward' composite wrapper in pybind_module.cpp:
        
        if hasattr(cuda_module, 'wtconv_forward'):
             # If you kept the composite wrapper
             out_cuda_tensor = cuda_module.wtconv_forward(x_cuda, w_cuda, stride, pad, groups, 0.5, 0.5)
        else:
             # If using atomic pipeline logic manually (Fallback test):
             # 1. DWT
             dwt_out = cuda_module.dwt_forward(x_cuda, 0.5)
             # 2. Conv
             conv_out = cuda_module.conv_forward(dwt_out, w_cuda, stride, pad)
             # 3. IDWT (Need to reshape for Interleaved IDWT? My atomic IDWT expects 4C interleaved)
             # conv_out is (N, 4C, H, W). This is exactly what IDWT expects.
             out_cuda_tensor = cuda_module.idwt_forward(conv_out, 0.5)

        # Move back to CPU
        out_cuda = out_cuda_tensor.cpu().numpy()
        
        diff_cuda = np.abs(out_ref - out_cuda).max()
        print(f"CUDA Max Diff: {diff_cuda:.8f}")
        
        if diff_cuda < 1e-5:
            print("✅ CUDA MATCH")
        else:
            print("❌ CUDA MISMATCH")
    else:
        print("\n[SKIP] CUDA module not loaded or GPU unavailable.")

if __name__ == "__main__":
    verify_repo_equivalence()