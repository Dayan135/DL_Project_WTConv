# bench_framework.py
#
# Generic benchmarking framework for multiple implementations (modules).
# UPDATED: Added OptimizedCudaAdapter for fused kernels.

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional, List, Tuple, Any

import numpy as np
import torch

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

# -----------------------------------------------------------------------------
# Timing
# -----------------------------------------------------------------------------
def time_cuda(fn: Callable[[], Any], iters: int, warmup: int) -> float:
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
    return start.elapsed_time(end) / iters  # ms/iter


def time_cpu(fn: Callable[[], Any], iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / iters  # ms/iter


# -----------------------------------------------------------------------------
# Benchmark spec
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class BenchConfig:
    batch: int = 16
    channels: int = 64
    height: int = 64
    width: int = 64
    kernel_size: int = 5
    wt_levels: int = 1
    wt_type: str = "db1"
    stride: int = 1
    dwt_scale: float = 0.5
    idwt_scale: float = 0.5


@dataclass
class BenchInputs:
    # Canonical tensors for reference (CUDA)
    x_cuda: torch.Tensor
    weights_cuda: List[torch.Tensor]
    groups: int


def build_inputs(cfg: BenchConfig, device: torch.device) -> Tuple[Any, BenchInputs]:
    """
    Builds reference layer and inputs.
    """
    if WTConv2d is None:
        raise RuntimeError("WTConv2d not available; cannot build weights from Reference repo.")

    x = torch.randn(cfg.batch, cfg.channels, cfg.height, cfg.width, device=device, dtype=torch.float32)

    ref = WTConv2d(
        cfg.channels,
        cfg.channels,
        kernel_size=cfg.kernel_size,
        wt_levels=cfg.wt_levels,
        wt_type=cfg.wt_type,
    ).to(device).eval()

    weights: List[torch.Tensor] = []
    for i in range(cfg.wt_levels):
        w = ref.wavelet_convs[i].weight.detach().contiguous().to(device)
        weights.append(w)

    out_c, in_c, _, _ = weights[0].shape
    groups = out_c if in_c == 1 else 1

    return ref, BenchInputs(
        x_cuda=x,
        weights_cuda=weights,
        groups=groups,
    )


# -----------------------------------------------------------------------------
# Adapter interface
# -----------------------------------------------------------------------------
class ImplAdapter:
    name: str = "base"
    kind: str = "cpu"  # "cpu" or "cuda"

    def available(self) -> Tuple[bool, str]:
        return False, "not implemented"

    def make_forward_fn(self, cfg: BenchConfig, ref_layer: Any, inp: BenchInputs) -> Callable[[], Any]:
        raise NotImplementedError

    def make_fwd_bwd_fn(self, cfg: BenchConfig, ref_layer: Any, inp: BenchInputs) -> Optional[Callable[[], Any]]:
        return None


# -----------------------------------------------------------------------------
# Adapter: Reference WTConv2d (PyTorch, CUDA, autograd)
# -----------------------------------------------------------------------------
class WTConv2dAdapter(ImplAdapter):
    name = "WTConv2d (Reference)"
    kind = "cuda"

    def available(self) -> Tuple[bool, str]:
        if WTConv2d is None:
            return False, "WTConv2d import failed"
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        return True, "ok"

    def make_forward_fn(self, cfg: BenchConfig, ref_layer: Any, inp: BenchInputs) -> Callable[[], Any]:
        x = inp.x_cuda
        layer = ref_layer

        def fn():
            return layer(x)

        return fn

    def make_fwd_bwd_fn(self, cfg: BenchConfig, ref_layer: Any, inp: BenchInputs) -> Optional[Callable[[], Any]]:
        x = inp.x_cuda.detach().clone().requires_grad_(True)
        layer = ref_layer.train()

        def fn():
            if x.grad is not None:
                x.grad.zero_()
            layer.zero_grad(set_to_none=True)
            y = layer(x)
            y.mean().backward()
            return None

        return fn


# -----------------------------------------------------------------------------
# Adapter: cpp_module (CPU Torch Tensors)
# -----------------------------------------------------------------------------
class CppAdapter(ImplAdapter):
    name = "cpp_module (CPU)"
    kind = "cpu"

    def available(self) -> Tuple[bool, str]:
        if cpp_module is None:
            return False, "cpp_module import failed"
        if not hasattr(cpp_module, "wtconv_forward"):
            return False, "cpp_module has no wtconv_forward"
        return True, "ok"

    def make_forward_fn(self, cfg: BenchConfig, ref_layer: Any, inp: BenchInputs) -> Callable[[], Any]:
        pad = cfg.kernel_size // 2
        stride = cfg.stride
        groups = inp.groups
        
        x_cpu = inp.x_cuda.detach().cpu().contiguous()
        w_cpu_list = [w.detach().cpu().contiguous() for w in inp.weights_cuda]

        def fn():
            return cpp_module.wtconv_forward(x_cpu, w_cpu_list, stride, pad, groups, cfg.dwt_scale, cfg.idwt_scale)

        return fn

    def make_fwd_bwd_fn(self, cfg: BenchConfig, ref_layer: Any, inp: BenchInputs) -> Optional[Callable[[], Any]]:
        if not hasattr(cpp_module, "wtconv_forward_save"): return None
        if not hasattr(cpp_module, "wtconv_backward"): return None

        pad = cfg.kernel_size // 2
        stride = cfg.stride
        groups = inp.groups
        
        x_cpu = inp.x_cuda.detach().cpu().contiguous()
        w_cpu_list = [w.detach().cpu().contiguous() for w in inp.weights_cuda]
        grad_out_cpu = torch.randn_like(x_cpu)

        def fn():
            y, saved = cpp_module.wtconv_forward_save(
                x_cpu, w_cpu_list, stride, pad, groups, cfg.dwt_scale, cfg.idwt_scale
            )
            gx, gw = cpp_module.wtconv_backward(
                saved, grad_out_cpu, w_cpu_list, groups
            )
            return None

        return fn


# -----------------------------------------------------------------------------
# Adapter: cuda_module (Baseline CUDA)
# -----------------------------------------------------------------------------
class CudaTorchAdapter(ImplAdapter):
    name = "cuda_module (Baseline)"
    kind = "cuda"

    def available(self) -> Tuple[bool, str]:
        if cuda_module is None:
            return False, "cuda_module import failed"
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        if not hasattr(cuda_module, "wtconv_forward"):
            return False, "cuda_module has no wtconv_forward"
        return True, "ok"

    def make_forward_fn(self, cfg: BenchConfig, ref_layer: Any, inp: BenchInputs) -> Callable[[], Any]:
        pad = cfg.kernel_size // 2
        stride = cfg.stride
        x = inp.x_cuda
        weights = inp.weights_cuda
        groups = inp.groups

        def fn():
            return cuda_module.wtconv_forward(x, weights, stride, pad, groups, cfg.dwt_scale, cfg.idwt_scale)

        return fn

    def make_fwd_bwd_fn(self, cfg: BenchConfig, ref_layer: Any, inp: BenchInputs) -> Optional[Callable[[], Any]]:
        if not hasattr(cuda_module, "wtconv_forward_save"): return None
        if not hasattr(cuda_module, "wtconv_backward"): return None

        pad = cfg.kernel_size // 2
        stride = cfg.stride
        groups = inp.groups

        x = inp.x_cuda.detach().clone().requires_grad_(True)
        weights = [w.detach().clone().requires_grad_(True) for w in inp.weights_cuda]

        class _Fn2(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x_, *w_list):
                y, saved = cuda_module.wtconv_forward_save(
                    x_, list(w_list), int(stride), int(pad), int(groups), float(cfg.dwt_scale), float(cfg.idwt_scale)
                )
                ctx.saved = saved
                ctx.w_list = w_list 
                return y

            @staticmethod
            def backward(ctx, grad_out):
                saved = ctx.saved
                w_list = list(ctx.w_list)
                gx, gw_list = cuda_module.wtconv_backward(
                    saved,
                    grad_out.contiguous(),
                    w_list,
                    int(groups),
                )
                return (gx, *gw_list)

        def fn():
            if x.grad is not None: x.grad.zero_()
            for w in weights:
                if w.grad is not None: w.grad.zero_()

            y = _Fn2.apply(x, *weights)
            y.mean().backward()
            return None

        return fn

# -----------------------------------------------------------------------------
# Adapter: optimized_cuda_module (Fused CUDA)
# -----------------------------------------------------------------------------
class OptimizedCudaAdapter(ImplAdapter):
    name = "opt_cuda_module (Fused)"
    kind = "cuda"

    def available(self) -> Tuple[bool, str]:
        if optimized_cuda_module is None:
            return False, "optimized_cuda_module import failed"
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        if not hasattr(optimized_cuda_module, "wtconv_forward"):
            return False, "opt_cuda has no wtconv_forward"
        return True, "ok"

    def make_forward_fn(self, cfg: BenchConfig, ref_layer: Any, inp: BenchInputs) -> Callable[[], Any]:
        pad = cfg.kernel_size // 2
        stride = cfg.stride
        x = inp.x_cuda
        weights = inp.weights_cuda
        groups = inp.groups

        def fn():
            # training=False by default (Inference Mode) - returns tuple, we take element 0
            out, _ = optimized_cuda_module.wtconv_forward(
                x, weights, stride, pad, groups, cfg.dwt_scale, cfg.idwt_scale, False
            )
            return out

        return fn

    def make_fwd_bwd_fn(self, cfg: BenchConfig, ref_layer: Any, inp: BenchInputs) -> Optional[Callable[[], Any]]:
        # Need backward function
        if not hasattr(optimized_cuda_module, "wtconv_backward"): return None

        pad = cfg.kernel_size // 2
        stride = cfg.stride
        groups = inp.groups

        x = inp.x_cuda.detach().clone().requires_grad_(True)
        weights = [w.detach().clone().requires_grad_(True) for w in inp.weights_cuda]

        class _Fn3(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x_, *w_list):
                # training=True (Training Mode)
                y, saved = optimized_cuda_module.wtconv_forward(
                    x_, list(w_list), int(stride), int(pad), int(groups), 
                    float(cfg.dwt_scale), float(cfg.idwt_scale), True
                )
                ctx.saved = saved
                ctx.w_list = w_list 
                return y

            @staticmethod
            def backward(ctx, grad_out):
                saved = ctx.saved
                w_list = list(ctx.w_list)
                gx, gw_list = optimized_cuda_module.wtconv_backward(
                    saved,
                    grad_out.contiguous(),
                    w_list,
                    int(groups),
                )
                return (gx, *gw_list)

        def fn():
            if x.grad is not None: x.grad.zero_()
            for w in weights:
                if w.grad is not None: w.grad.zero_()

            y = _Fn3.apply(x, *weights)
            y.mean().backward()
            return None

        return fn


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------
def default_registry() -> List[ImplAdapter]:
    return [
        WTConv2dAdapter(),
        #CppAdapter(),
        CudaTorchAdapter(),
        OptimizedCudaAdapter(), # <--- ADDED
    ]


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
@dataclass
class ResultRow:
    name: str
    available: bool
    reason: str
    forward_ms: Optional[float]
    fwd_bwd_ms: Optional[float]
    kind: str


def run_bench(
    adapters: List[ImplAdapter],
    cfg: BenchConfig,
    iters: int,
    warmup: int,
) -> List[ResultRow]:
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    ref_layer, inp = build_inputs(cfg, device=device)

    rows: List[ResultRow] = []

    for ad in adapters:
        ok, reason = ad.available()
        if not ok:
            rows.append(ResultRow(ad.name, False, reason, None, None, ad.kind))
            continue

        # forward
        fwd_fn = ad.make_forward_fn(cfg, ref_layer, inp)
        if ad.kind == "cuda":
            fwd_ms = time_cuda(fwd_fn, iters=iters, warmup=warmup)
        else:
            fwd_ms = time_cpu(fwd_fn, iters=iters, warmup=max(1, warmup // 5))

        # fwd+bwd (optional)
        fwd_bwd_fn = ad.make_fwd_bwd_fn(cfg, ref_layer, inp)
        if fwd_bwd_fn is None:
            fwd_bwd_ms = None
        else:
            if ad.kind != "cuda":
                fwd_bwd_ms = time_cpu(fwd_bwd_fn, iters=iters, warmup=max(1, warmup // 5))
            else:
                fwd_bwd_ms = time_cuda(fwd_bwd_fn, iters=iters, warmup=warmup)

        rows.append(ResultRow(ad.name, True, "ok", fwd_ms, fwd_bwd_ms, ad.kind))

    return rows


def print_results(rows: List[ResultRow]) -> None:
    print("\n=== Results ===")
    for r in rows:
        if not r.available:
            print(f"- {r.name}: SKIP ({r.reason})")
            continue

        fwd = "N/A" if r.forward_ms is None else f"{r.forward_ms:.4f} ms/iter"
        bwd = "N/A" if r.fwd_bwd_ms is None else f"{r.fwd_bwd_ms:.4f} ms/iter"
        print(f"- {r.name:24s} | forward: {fwd:16s} | fwd+bwd: {bwd:16s} | kind={r.kind}")


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--kernel", type=int, default=5)
    p.add_argument("--levels", type=int, default=1)
    args = p.parse_args()

    cfg = BenchConfig(
        batch=args.batch,
        channels=args.channels,
        height=args.height,
        width=args.width,
        kernel_size=args.kernel,
        wt_levels=args.levels,
    )

    print("\n=== Benchmark Framework ===")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Shape:  N={cfg.batch}, C={cfg.channels}, H={cfg.height}, W={cfg.width}")
    print(f"K:      {cfg.kernel_size}, wt_levels={cfg.wt_levels}")
    print(f"iters={args.iters}, warmup={args.warmup}")

    adapters = default_registry()
    rows = run_bench(adapters, cfg, iters=args.iters, warmup=args.warmup)
    print_results(rows)

    # Speedups vs reference
    ref = next((r for r in rows if r.name.startswith("WTConv2d") and r.available), None)
    
    if ref is not None and ref.forward_ms is not None:
        print("\n=== Speedup vs WTConv2d (forward) ===")
        for r in rows:
            if not r.available or r.forward_ms is None: continue
            if r.kind != ref.kind:
                # CPU vs CUDA comparisons are valid here
                pass
            print(f"- {r.name}: {ref.forward_ms / r.forward_ms:.2f}x")

    if ref is not None and ref.fwd_bwd_ms is not None:
        print("\n=== Speedup vs WTConv2d (forward+backward) ===")
        for r in rows:
            if not r.available or r.fwd_bwd_ms is None: continue
            print(f"- {r.name}: {ref.fwd_bwd_ms / r.fwd_bwd_ms:.2f}x")


if __name__ == "__main__":
    main()