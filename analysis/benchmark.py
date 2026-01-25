import os
import sys
import torch
import time
import argparse

# -----------------------------
# 0) Path Setup & Compilation
# -----------------------------
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(ANALYSIS_DIR)
import compile_utils

# Check for --no-build flag *before* we do anything else
if "--no-build" in sys.argv:
    os.environ["SKIP_COMPILE"] = "1"
    print("[INFO] Compilation disabled via --no-build")
else:
    print("--- Triggering Auto-Compilation ---")
    compile_utils.compile_all()

# -----------------------------
# 1) Paths & Imports
# -----------------------------
ref_repo_path = os.path.join(HERE, "Reference")
cpp_source_path = os.path.join(HERE, "cpp_source")
cuda_source_path = os.path.join(HERE, "cuda_source")
opt_cuda_source_path = os.path.join(HERE, "optimized_cuda_source")

sys.path.insert(0, ref_repo_path)
sys.path.append(cpp_source_path)
sys.path.append(cuda_source_path)
sys.path.append(opt_cuda_source_path)

# --- Load Modules ---
modules = {}
try: import cpp_module; modules['cpp'] = cpp_module
except: pass
try: import cuda_module; modules['cuda'] = cuda_module
except: pass
try: import optimized_cuda_module; modules['opt_cuda'] = optimized_cuda_module
except: pass

try:
    from wtconv.wtconv2d import WTConv2d
except ImportError as e:
    print(f"[ERROR] Could not import WTConv2d: {e}")
    sys.exit(1)

# -----------------------------
# 2) Timing Utils
# -----------------------------
def time_cuda(fn, iters=100, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters): fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters

def time_cpu(fn, iters=20, warmup=2):
    for _ in range(warmup): fn()
    start = time.time()
    for _ in range(iters): fn()
    return ((time.time() - start) * 1000) / iters

def make_fn(mod, x, w, s, p, g, ds, ids, use_cuda=True):
    if use_cuda:
        return lambda: mod.wtconv_forward(x, w, s, p, g, ds, ids)
    else:
        # Move to CPU once
        x_c = x.cpu().contiguous()
        w_c = [ww.cpu().contiguous() for ww in w]
        return lambda: mod.wtconv_forward(x_c, w_c, s, p, g, ds, ids)

# -----------------------------
# 3) Main Benchmark
# -----------------------------
def run_benchmark(
    batch=16, 
    channels=64, 
    height=64, 
    width=64, 
    kernel_size=5, 
    wt_levels=2, 
    iterations=100
):
    print("\n=== Benchmark: Baseline vs Optimized ===")
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available.")
        return

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    
    stride = 1
    pad = kernel_size // 2

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Input:  N={batch}, C={channels}, H={height}, W={width}")
    print(f"Config: K={kernel_size}, L={wt_levels}")

    # Data
    x = torch.randn(batch, channels, height, width, device=device)
    ref = WTConv2d(channels, channels, kernel_size=kernel_size, wt_levels=wt_levels, wt_type="db1").to(device).eval()
    weights = [ref.wavelet_convs[i].weight.detach().contiguous().to(device) for i in range(wt_levels)]
    
    out_c, in_c, _, _ = weights[0].shape
    groups = out_c if in_c == 1 else 1
    print(f"Groups: {groups}")

    results = {}

    # 1. Reference
    print("Benchmarking Reference...")
    ref_fn = lambda: ref(x)
    results['PyTorch Ref'] = time_cuda(ref_fn, iters=iterations)

    # # 2. C++
    # if 'cpp' in modules:
    #     print("Benchmarking C++ (CPU)...")
    #     # Fewer iters for CPU
    #     fn = make_fn(modules['cpp'], x, weights, stride, pad, groups, 0.5, 0.5, False)
    #     # Dynamic CPU iters based on total iterations requested
    #     cpu_iters = max(5, iterations // 5)
    #     results['C++ (CPU)'] = time_cpu(fn, iters=cpu_iters)

    # 3. Baseline CUDA
    if 'cuda' in modules:
        print("Benchmarking Baseline CUDA...")
        fn = make_fn(modules['cuda'], x, weights, stride, pad, groups, 0.5, 0.5, True)
        results['Baseline CUDA'] = time_cuda(fn, iters=iterations)

    # 4. Optimized CUDA
    if 'opt_cuda' in modules:
        print("Benchmarking Optimized CUDA...")
        fn = make_fn(modules['opt_cuda'], x, weights, stride, pad, groups, 0.5, 0.5, True)
        results['Optimized CUDA'] = time_cuda(fn, iters=iterations)

    # Report
    print("\n--- Final Results (ms/iter) ---")
    ref_ms = results.get('PyTorch Ref')
    
    for name, ms in results.items():
        ratio_str = ""
        if ref_ms:
            factor = ref_ms / ms
            ratio_str = f"({factor:.2f}x vs Ref)"
            
        print(f"{name:<15}: {ms:.4f} ms {ratio_str}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WTConv2d Benchmark")
    
    # Flags
    parser.add_argument("--no-build", action="store_true", help="Skip compilation step")
    
    # Config
    parser.add_argument("-b", "--batch", type=int, default=16, help="Batch size")
    parser.add_argument("-c", "--channels", type=int, default=64, help="Input/Output Channels")
    parser.add_argument("--height", type=int, default=64, help="Input Height")
    parser.add_argument("--width", type=int, default=64, help="Input Width")
    parser.add_argument("-k", "--kernel-size", type=int, default=5, help="Kernel Size")
    parser.add_argument("-l", "--levels", type=int, default=2, help="Wavelet Levels")
    parser.add_argument("-i", "--iterations", type=int, default=100, help="Benchmark Iterations")

    args = parser.parse_args()
    
    # 1. Run Config from Command Line
    run_benchmark(
        batch=args.batch,
        channels=args.channels,
        height=args.height,
        width=args.width,
        kernel_size=args.kernel_size,
        wt_levels=args.levels,
        iterations=args.iterations
    )

    # 2. Additional Hardcoded Tests
    print("\n" + "="*40 + "\n[AUTO] Running Extra Sizes...\n" + "="*40)
    
    # Large Image, Small Kernel
    run_benchmark(batch=64, channels=5, height=224, width=224, kernel_size=3, wt_levels=3, iterations=50)
    
    # Large Image, Larger Kernel
    run_benchmark(batch=64, channels=5, height=224, width=224, kernel_size=7, wt_levels=3, iterations=50)

    # Large Image, Small Kernel
    run_benchmark(batch=64, channels=5, height=224, width=224, kernel_size=3, wt_levels=5, iterations=50)
    
    # Large Image, Larger Kernel
    run_benchmark(batch=64, channels=5, height=224, width=224, kernel_size=7, wt_levels=5, iterations=50)