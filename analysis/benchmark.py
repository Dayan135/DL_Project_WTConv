import os
import sys
import torch
import time

# -----------------------------
# 0) Paths
# -----------------------------
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ref_repo_path = os.path.join(HERE, "Reference")
cpp_source_path = os.path.join(HERE, "cpp_source")
cuda_source_path = os.path.join(HERE, "cuda_source")

if not os.path.exists(ref_repo_path):
    print(f"[ERROR] Reference path not found: {ref_repo_path}")
    sys.exit(1)

sys.path.insert(0, ref_repo_path)
sys.path.append(cpp_source_path)
sys.path.append(cuda_source_path)

# -----------------------------
# 1) Imports
# -----------------------------
try:
    from wtconv.wtconv2d import WTConv2d
except ImportError as e:
    print(f"[ERROR] Could not import WTConv2d from Reference: {e}")
    sys.exit(1)

try:
    import cpp_module
except ImportError as e:
    print(f"[ERROR] Could not import cpp_module: {e}")
    sys.exit(1)

cuda_module = None
try:
    import cuda_module as _cuda_module
    cuda_module = _cuda_module
    print("[INFO] cuda_module loaded successfully.")
except ImportError:
    print("[WARN] cuda_module could not be imported.")
    cuda_module = None


# -----------------------------
# 2) Timing Utils
# -----------------------------
def _time_cuda(fn, iters: int, warmup: int = 20) -> float:
    """ Returns: avg milliseconds per iteration for fn() on GPU. """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters

def _time_cpu(fn, iters: int, warmup: int = 5) -> float:
    """ Returns: avg milliseconds per iteration for fn() on CPU. """
    for _ in range(warmup):
        fn()
    
    start = time.time()
    for _ in range(iters):
        fn()
    end = time.time()
    
    total_ms = (end - start) * 1000
    return total_ms / iters

# -----------------------------
# 3) Adapters
# -----------------------------
def _make_wtconv2d_fn(layer: WTConv2d, x: torch.Tensor):
    def fn():
        return layer(x)
    return fn

def _make_cpp_module_fn(x: torch.Tensor, weights: list[torch.Tensor], stride: int, pad: int, groups: int,
                        dwt_scale: float, idwt_scale: float):
    # Move inputs to CPU once before benchmarking
    x_cpu = x.cpu().contiguous()
    w_cpu = [w.cpu().contiguous() for w in weights]
    
    def fn():
        return cpp_module.wtconv_forward(x_cpu, w_cpu, stride, pad, groups, dwt_scale, idwt_scale)
    return fn

def _make_cuda_module_fn(x: torch.Tensor, weights: list[torch.Tensor], stride: int, pad: int, groups: int,
                          dwt_scale: float, idwt_scale: float):
    def fn():
        return cuda_module.wtconv_forward(x, weights, stride, pad, groups, dwt_scale, idwt_scale)
    return fn


# -----------------------------
# 4) Main Benchmark Function
# -----------------------------
def run_benchmark(
    batch=16,
    channels=64,
    height=64,
    width=64,
    kernel_size=5,
    wt_levels=1,
    iterations=100,
    dwt_scale=0.5,
    idwt_scale=0.5
):
    print("\n=== Benchmark: WTConv2d vs cpp_module (CPU) vs cuda_module (GPU) ===")

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available. Run on a GPU node.")
        return

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    # Derived Config
    stride = 1
    pad = kernel_size // 2

    # --- Print Stats ---
    print(f"Device:   {torch.cuda.get_device_name(0)}")
    print(f"Input:    (N={batch}, C={channels}, H={height}, W={width})")
    print(f"Kernel:   K={kernel_size}, Levels={wt_levels}")
    print(f"Scales:   DWT={dwt_scale}, IDWT={idwt_scale}")
    print(f"Iters:    {iterations}")

    # Input
    x = torch.randn(batch, channels, height, width, device=device, dtype=torch.float32)

    # Reference layer
    ref = WTConv2d(channels, channels, kernel_size=kernel_size, wt_levels=wt_levels, wt_type="db1").to(device).eval()

    # Pull weights
    weights = []
    for i in range(wt_levels):
        w = ref.wavelet_convs[i].weight.detach().contiguous().to(device)
        weights.append(w)

    # Infer groups
    out_c, in_c, _, _ = weights[0].shape
    groups = out_c if in_c == 1 else 1
    print(f"Groups:   {groups} ({'Depthwise' if groups > 1 else 'Dense'})")

    # Build functions
    wt_fn = _make_wtconv2d_fn(ref, x)
    
    # 1. Reference Time (PyTorch GPU)
    print("\nBenchmarking Reference (PyTorch GPU)...")
    wt_ms = _time_cuda(wt_fn, iters=iterations, warmup=10)
    print(f" -> {wt_ms:.4f} ms/iter")

    # 2. C++ Time (CPU)
    print("Benchmarking cpp_module (CPU)...")
    cpp_fn = _make_cpp_module_fn(x, weights, stride, pad, groups, dwt_scale, idwt_scale)
    # Reduce iters for CPU because it is very slow
    cpu_iters = max(5, iterations // 5)
    cpp_ms = _time_cpu(cpp_fn, iters=cpu_iters, warmup=2)
    print(f" -> {cpp_ms:.4f} ms/iter")

    # 3. CUDA Time (GPU)
    cuda_ms = None
    if cuda_module is not None:
        print("Benchmarking cuda_module (GPU)...")
        cuda_fn = _make_cuda_module_fn(x, weights, stride, pad, groups, dwt_scale, idwt_scale)
        cuda_ms = _time_cuda(cuda_fn, iters=iterations, warmup=10)
        print(f" -> {cuda_ms:.4f} ms/iter")

    # Results
    print("\n--- Final Results ---")
    print(f"WTConv2d (Ref): {wt_ms:.4f} ms")
    print(f"cpp_module:     {cpp_ms:.4f} ms (CPU)")
    if cuda_ms:
        print(f"cuda_module:    {cuda_ms:.4f} ms (GPU)")
        print(f"\n>> CUDA Kernel Speedup vs PyTorch: {wt_ms / cuda_ms:.2f}x")

if __name__ == "__main__":
    # You can change defaults here or import run_benchmark in another script
    run_benchmark(wt_levels=5)