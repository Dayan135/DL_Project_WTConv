# benchmark_three_impls.py
#
# Benchmarks (CUDA):
#   1) Reference PyTorch layer: WTConv2d (from Reference repo)
#   2) cpp_module extension      (your custom C++/CUDA extension)
#   3) cuda_module extension     (another C++/CUDA extension, if available)
#
# Usage:
#   python benchmark_three_impls.py
#
# Notes:
# - This file assumes cpp_module exposes: wtconv_forward(x, weights_list, stride, pad, groups, dwt_scale, idwt_scale)
# - It assumes cuda_module exposes the same API. If it differs, edit the adapter in _make_cuda_module_fn().

import os
import sys
import time
import torch
import numpy as np

# -----------------------------
# 0) Paths
# -----------------------------
HERE = os.path.dirname(os.path.abspath(__file__))

ref_repo_path = os.path.join(HERE, "Reference")
cpp_source_path = os.path.join(HERE, "cpp_source")
cuda_source_path = os.path.join(HERE, "cuda_source")

if not os.path.exists(ref_repo_path):
    print(f"[ERROR] Reference path not found: {ref_repo_path}")
    sys.exit(1)

sys.path.insert(0, ref_repo_path)      # Priority to Reference
sys.path.append(cpp_source_path)       # Where cpp extension lives/builds
sys.path.append(cuda_source_path)      # Where cuda extension lives/builds


# -----------------------------
# 1) Imports
# -----------------------------
# Reference implementation (PyTorch)
try:
    from wtconv.wtconv2d import WTConv2d
except ImportError as e:
    print(f"[ERROR] Could not import WTConv2d from Reference: {e}")
    sys.exit(1)

# cpp_module (required)
try:
    import cpp_module
except ImportError as e:
    print(f"[ERROR] Could not import cpp_module: {e}")
    print("Compile it first (setup.py / pip install -e .).")
    sys.exit(1)

# cuda_module (optional)
cuda_module = None
try:
    import cuda_module as _cuda_module
    cuda_module = _cuda_module
except ImportError:
    cuda_module = None


# -----------------------------
# 2) Timing utils
# -----------------------------
def _time_cuda(fn, iters: int, warmup: int = 20) -> float:
    """
    Returns: avg milliseconds per iteration for fn() on CUDA.
    """
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
    total_ms = start.elapsed_time(end)
    return total_ms / iters


def _time_cpu(fn, iters: int, warmup: int = 5) -> float:
    """
    Returns: avg milliseconds per iteration for fn() on CPU.
    """
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / iters  # ms/iter


# -----------------------------
# 3) Adapters
# -----------------------------
def _make_wtconv2d_fn(layer: WTConv2d, x: torch.Tensor):
    def fn():
        return layer(x)
    return fn


def _make_cpp_module_fn(x: torch.Tensor, weights: list[torch.Tensor], stride: int, pad: int, groups: int,
                        dwt_scale: float, idwt_scale: float):
    """
    cpp_module is a CPU numpy extension.
    Convert tensors to numpy once (outside the timing loop).
    """
    x_np = x.detach().cpu().numpy().astype(np.float32, copy=False)
    w_np = weights[0].detach().cpu().numpy().astype(np.float32, copy=False)
    
    def fn():
        return cpp_module.wtconv_forward(x_np, w_np, stride, pad, groups, dwt_scale, idwt_scale)
    return fn


def _make_cuda_module_fn(x: torch.Tensor, weights: list[torch.Tensor], stride: int, pad: int, groups: int,
                         dwt_scale: float, idwt_scale: float):
    """
    cuda_module is a CUDA torch extension.
    Pass torch tensors directly (weights is already a list, don't nest it).
    """
    def fn():
        return cuda_module.wtconv_forward(x, weights, stride, pad, groups, dwt_scale, idwt_scale)
    return fn


# -----------------------------
# 4) Main benchmark
# -----------------------------
def run():
    print("\n=== Benchmark: WTConv2d vs cpp_module vs cuda_module (CUDA) ===")

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available. Run on a GPU node.")
        return

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    # --- Config (edit freely) ---
    BATCH = 16
    CHANNELS = 64
    HEIGHT = 64
    WIDTH = 64
    KERNEL_SIZE = 5
    WT_LEVELS = 1
    ITERATIONS = 200

    stride = 1
    pad = KERNEL_SIZE // 2

    dwt_scale = 0.5
    idwt_scale = 0.5

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Shape:  N={BATCH}, C={CHANNELS}, H={HEIGHT}, W={WIDTH}")
    print(f"K:      {KERNEL_SIZE}, wt_levels={WT_LEVELS}, iters={ITERATIONS}")
    print(f"Scales: dwt={dwt_scale}, idwt={idwt_scale}")

    # Input
    x = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH, device=device, dtype=torch.float32)

    # Reference layer (WTConv2d)
    ref = WTConv2d(
        CHANNELS, CHANNELS,
        kernel_size=KERNEL_SIZE,
        wt_levels=WT_LEVELS,
        wt_type="db1",
    ).to(device).eval()

    # Pull weights for extensions:
    # In the Reference repo, wavelet_convs is a list per level.
    weights = []
    for i in range(WT_LEVELS):
        w = ref.wavelet_convs[i].weight.detach().contiguous().to(device)
        weights.append(w)

    # Infer groups (depthwise if in_c == 1)
    # Using level-0 weight to decide; should match all levels.
    out_c, in_c, _, _ = weights[0].shape
    groups = out_c if in_c == 1 else 1
    print(f"Groups inferred: {groups}")

    # Build benchmark functions
    wt_fn = _make_wtconv2d_fn(ref, x)
    cpp_fn = _make_cpp_module_fn(x, weights, stride, pad, groups, dwt_scale, idwt_scale)

    cuda_fn = None
    if cuda_module is not None:
        cuda_fn = _make_cuda_module_fn(x, weights, stride, pad, groups, dwt_scale, idwt_scale)
        print(f"cuda_module loaded from: {cuda_module.__file__}")
    else:
        print("cuda_module: NOT FOUND (will skip). If you expected it, compile/install it and ensure import works.")

    # Run (no grad)
    with torch.no_grad():
        # WTConv2d (Reference) - CUDA
        wt_ms = _time_cuda(wt_fn, iters=ITERATIONS, warmup=30)

        # cpp_module - CPU (cannot use CUDA events for CPU code)
        cpp_ms = _time_cpu(cpp_fn, iters=ITERATIONS, warmup=10)

        # cuda_module - CUDA
        cuda_ms = None
        if cuda_fn is not None:
            cuda_ms = _time_cuda(cuda_fn, iters=ITERATIONS, warmup=30)

    # Print results
    print("\n--- Results (Forward only) ---")
    print(f"WTConv2d (Reference): {wt_ms:.4f} ms/iter (CUDA)")
    print(f"cpp_module:           {cpp_ms:.4f} ms/iter (CPU)")

    if cuda_ms is not None:
        print(f"cuda_module:          {cuda_ms:.4f} ms/iter (CUDA)")

    # Relative speeds vs WTConv2d
    print("\n--- Speedup vs WTConv2d (CUDA baseline) ---")
    print(f"cpp_module (CPU):  {wt_ms / cpp_ms:.5f}x (note: CPU vs CUDA, not a fair comparison)")
    if cuda_ms is not None:
        print(f"cuda_module (CUDA): {wt_ms / cuda_ms:.2f}x")


if __name__ == "__main__":
    run()
