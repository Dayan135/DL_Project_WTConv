# bench_framework.py
#
# Generic benchmarking framework for multiple implementations (modules),
# designed to make it trivial to add more modules later.
#
# Supports:
# - PyTorch modules with autograd on CUDA/CPU (e.g., WTConv2d)
# - Torch CUDA extensions (torch.Tensor I/O) with:
#     a) autograd support, OR
#     b) explicit forward/backward functions, OR
#     c) forward-only (backward skipped)
# - NumPy CPU extensions (numpy.ndarray I/O) forward-only (backward skipped)
#
# This version is IMPLEMENTED to match the C++/pybind API we defined:
#   - cuda_module.wtconv_forward(x, weights, stride, pad, groups, dwt_scale, idwt_scale) -> y
#   - cuda_module.wtconv_forward_save(x, weights, stride, pad, groups, dwt_scale, idwt_scale) -> (y, saved)
#   - cuda_module.wtconv_backward(saved, grad_out, weights, groups) -> (grad_x, grad_weights_list)
#
# Usage:
#   python bench_framework.py --iters 200 --warmup 30

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional, List, Tuple, Any

import numpy as np
import torch


# -----------------------------------------------------------------------------
# Paths / imports (repo-specific)
# -----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REF_REPO_PATH = os.path.join(HERE, "Reference")
CPP_SOURCE_PATH = os.path.join(HERE, "cpp_source")
CUDA_SOURCE_PATH = os.path.join(HERE, "cuda_source")

if os.path.exists(REF_REPO_PATH):
    sys.path.insert(0, REF_REPO_PATH)
if os.path.exists(CPP_SOURCE_PATH):
    sys.path.append(CPP_SOURCE_PATH)
if os.path.exists(CUDA_SOURCE_PATH):
    sys.path.append(CUDA_SOURCE_PATH)

try:
    from wtconv.wtconv2d import WTConv2d
except Exception as e:
    WTConv2d = None
    print(f"[WARN] WTConv2d not importable: {e}")

try:
    import cpp_module
except Exception as e:
    cpp_module = None
    print(f"[WARN] cpp_module not importable: {e}")

try:
    import cuda_module
except Exception as e:
    cuda_module = None
    print(f"[WARN] cuda_module not importable: {e}")


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

    # CPU numpy versions (for numpy extensions)
    x_np: np.ndarray
    w_np: np.ndarray


def build_inputs(cfg: BenchConfig, device: torch.device) -> Tuple[Any, BenchInputs]:
    """
    Builds:
      - ref_layer (WTConv2d) on the given device (usually CUDA) for reference timing
      - canonical input tensor (CUDA) and weights list (CUDA)
      - numpy CPU views for cpp_module forward-only
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

    x_np = x.detach().cpu().numpy().astype(np.float32, copy=False)
    w_np = weights[0].detach().cpu().numpy().astype(np.float32, copy=False)

    return ref, BenchInputs(
        x_cuda=x,
        weights_cuda=weights,
        groups=groups,
        x_np=x_np,
        w_np=w_np,
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
# Adapter: cpp_module (NumPy CPU forward-only)
# -----------------------------------------------------------------------------
class CppNumpyAdapter(ImplAdapter):
    name = "cpp_module (NumPy CPU)"
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
        x_np = inp.x_np
        w_np = inp.w_np

        def fn():
            return cpp_module.wtconv_forward(x_np, w_np, stride, pad, groups, cfg.dwt_scale, cfg.idwt_scale)

        return fn

    def make_fwd_bwd_fn(self, cfg: BenchConfig, ref_layer: Any, inp: BenchInputs) -> Optional[Callable[[], Any]]:
        return None


# -----------------------------------------------------------------------------
# Adapter: cuda_module (Torch CUDA extension)
# -----------------------------------------------------------------------------
class CudaTorchAdapter(ImplAdapter):
    name = "cuda_module (Torch CUDA)"
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
            # Forward-only fast path (no saved tensors)
            return cuda_module.wtconv_forward(x, weights, stride, pad, groups, cfg.dwt_scale, cfg.idwt_scale)

        return fn

    def make_fwd_bwd_fn(self, cfg: BenchConfig, ref_layer: Any, inp: BenchInputs) -> Optional[Callable[[], Any]]:
        """
        Uses the explicit save/backward API:
          - wtconv_forward_save(...) -> (y, saved)
          - wtconv_backward(saved, grad_out, weights, groups) -> (gx, gw_list)
        """
        if not hasattr(cuda_module, "wtconv_forward_save"):
            return None
        if not hasattr(cuda_module, "wtconv_backward"):
            return None

        pad = cfg.kernel_size // 2
        stride = cfg.stride
        groups = inp.groups

        # Separate tensors so we don't mutate reference weights
        x = inp.x_cuda.detach().clone().requires_grad_(True)
        weights = [w.detach().clone().requires_grad_(True) for w in inp.weights_cuda]

        class _Fn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x_, *w_list):
                # IMPORTANT: we want saved tensors only when grad enabled; C++ already gates this.
                y, saved = cuda_module.wtconv_forward_save(
                    x_, list(w_list), int(stride), int(pad), int(groups), float(cfg.dwt_scale), float(cfg.idwt_scale)
                )
                ctx.saved = saved
                ctx.w_list_len = len(w_list)
                return y

            @staticmethod
            def backward(ctx, grad_out):
                # If forward ran in a context where saving was disabled, saved will be empty.
                saved = ctx.saved
                if not saved:
                    raise RuntimeError("cuda_module forward did not save tensors; cannot run backward.")

                # We must return grads for inputs: (x_, *w_list)
                # Our C++ backward expects: (saved, grad_out, weights, groups)
                # and returns (grad_x, grad_weights_list)
                # NOTE: Weights are passed from outer closure, but backward must match current w_list values.
                # PyTorch gives us no direct access to w_list here (unless we saved them), but we don't need to:
                # we can reconstruct from inputs by saving them, OR we can assume weights are constant.
                #
                # Correct approach: save w_list in ctx as tensors.
                # We'll do that by additionally saving them as attributes (not via save_for_backward to avoid copies).
                raise RuntimeError("Internal error: backward requires access to weights; see wrapper below.")

        # Correct wrapper that saves weights references
        class _Fn2(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x_, *w_list):
                y, saved = cuda_module.wtconv_forward_save(
                    x_, list(w_list), int(stride), int(pad), int(groups), float(cfg.dwt_scale), float(cfg.idwt_scale)
                )
                ctx.saved = saved
                ctx.w_list = w_list  # references to tensors
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
                # return grad for (x_, *w_list)
                return (gx, *gw_list)

        def fn():
            # zero grads
            if x.grad is not None:
                x.grad.zero_()
            for w in weights:
                if w.grad is not None:
                    w.grad.zero_()

            y = _Fn2.apply(x, *weights)
            y.mean().backward()
            return None

        return fn


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------
def default_registry() -> List[ImplAdapter]:
    return [
        WTConv2dAdapter(),
        CppNumpyAdapter(),
        CudaTorchAdapter(),
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
    p.add_argument("--iters", type=int, default=200)
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

    # Speedups vs reference (forward only, CUDA-only)
    ref = next((r for r in rows if r.name.startswith("WTConv2d") and r.available), None)
    if ref is not None and ref.forward_ms is not None:
        print("\n=== Speedup vs WTConv2d (forward) ===")
        for r in rows:
            if not r.available or r.forward_ms is None:
                continue
            if r.kind != ref.kind:
                print(f"- {r.name}: N/A (kind mismatch: {r.kind} vs {ref.kind})")
                continue
            print(f"- {r.name}: {ref.forward_ms / r.forward_ms:.2f}x")

    if ref is not None and ref.fwd_bwd_ms is not None:
        print("\n=== Speedup vs WTConv2d (forward+backward) ===")
        for r in rows:
            if not r.available or r.fwd_bwd_ms is None:
                continue
            if r.kind != ref.kind:
                print(f"- {r.name}: N/A (kind mismatch: {r.kind} vs {ref.kind})")
                continue
            print(f"- {r.name}: {ref.fwd_bwd_ms / r.fwd_bwd_ms:.2f}x")


if __name__ == "__main__":
    main()
