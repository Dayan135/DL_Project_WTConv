import sys
import os
import time
import torch
import numpy as np

# 1. Setup Paths
repo_path = os.path.join(os.path.dirname(__file__), 'Reference')
if not os.path.exists(repo_path):
    print(f"[ERROR] Reference path not found: {repo_path}")
    sys.exit(1)
sys.path.insert(0, repo_path) # Priority to Reference
sys.path.append(os.path.join(os.path.dirname(__file__), 'cpp_source'))

# 2. Imports
try:
    import cpp_module
except ImportError:
    print("[ERROR] Could not import cpp_module. Compile it first.")
    sys.exit(1)

try:
    from wtconv.wtconv2d import WTConv2d
    print("[INFO] Imported WTConv2d from Reference.")
except ImportError as e:
    print(f"[ERROR] Could not import WTConv2d: {e}")
    sys.exit(1)

def run_benchmark():
    print("\n=== Performance Benchmark: BGU Repo vs C++ Kernel ===")
    
    # --- Configuration ---
    # Using specific sizes to test load
    BATCH = 16
    CHANNELS = 64
    HEIGHT = 64
    WIDTH = 64
    KERNEL_SIZE = 5
    ITERATIONS = 50
    
    # Instantiate Reference
    # Note: BGU uses depthwise logic (groups=4*C internally)
    ref_layer = WTConv2d(CHANNELS, CHANNELS, kernel_size=KERNEL_SIZE, wt_levels=1, wt_type='db1')
    ref_layer.eval()
    
    # Inputs
    x_torch = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
    
    # Extract Weights and Setup C++ Args
    # We must grab the internal conv weight
    weight_torch = ref_layer.wavelet_convs[0].weight.detach()
    
    out_c, in_c, k, _ = weight_torch.shape
    # Groups detection logic
    if in_c == 1:
        groups = out_c 
    else:
        groups = 1

    x_np = x_torch.numpy()
    w_np = weight_torch.numpy()
    stride = 1
    pad = KERNEL_SIZE // 2
    
    print(f"Config: Batch={BATCH}, C={CHANNELS}, H={HEIGHT}, W={WIDTH}, Groups={groups}")
    print(f"Running {ITERATIONS} iterations...")

    # --- 1. PyTorch Benchmark ---
    # We run the FULL layer (including scale/residual) because that is "The Repo Implementation"
    # To be strictly fair to kernel vs kernel, we could disable scales, but PyTorch overhead is low.
    
    # Warmup
    for _ in range(5):
        _ = ref_layer(x_torch)
        
    start_time = time.time()
    for _ in range(ITERATIONS):
        _ = ref_layer(x_torch)
    torch_time = time.time() - start_time
    
    # --- 2. C++ Benchmark ---
    # Warmup
    for _ in range(5):
        _ = cpp_module.wtconv_forward(x_np, w_np, stride, pad, groups, 0.5, 0.5)

    start_time = time.time()
    for _ in range(ITERATIONS):
        # Note: We pass 0.5, 0.5 scales to match correctness
        _ = cpp_module.wtconv_forward(x_np, w_np, stride, pad, groups, 0.5, 0.5)
    cpp_time = time.time() - start_time

    # --- Results ---
    avg_torch = (torch_time / ITERATIONS) * 1000
    avg_cpp = (cpp_time / ITERATIONS) * 1000
    
    print("\n--- Results (Forward Pass) ---")
    print(f"PyTorch (Repo):  {avg_torch:.4f} ms/iter")
    print(f"C++ Kernel:      {avg_cpp:.4f} ms/iter")
    
    if avg_cpp < avg_torch:
        print(f"\n>> C++ is {avg_torch/avg_cpp:.2f}x FASTER 🚀")
    else:
        print(f"\n>> C++ is {avg_cpp/avg_torch:.2f}x SLOWER 🐢")
        print("(Note: PyTorch uses AVX2/MKL optimizations. Ensure your C++ is compiled with -O3 and OpenMP)")

if __name__ == "__main__":
    run_benchmark()