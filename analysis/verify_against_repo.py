import sys
import os
import torch
import numpy as np
from pathlib import Path

# --- 1. Dynamic Path Setup & Compilation ---
# HERE = Project Root (DL_Project)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure we can find compile_utils (in analysis folder)
sys.path.append(ANALYSIS_DIR) 
import compile_utils

# --- TRIGGER COMPILATION BEFORE IMPORTS ---
print("--- Triggering Auto-Compilation ---")
compile_utils.compile_all()
# ------------------------------------------

ref_repo_path = os.path.join(HERE, "Reference")
cpp_source_path = os.path.join(HERE, "cpp_source")
cuda_source_path = os.path.join(HERE, "cuda_source")
opt_cuda_source_path = os.path.join(HERE, "optimized_cuda_source")

if not os.path.exists(ref_repo_path):
    print(f"[ERROR] Reference path not found: {ref_repo_path}")
    sys.exit(1)

sys.path.insert(0, ref_repo_path)
sys.path.append(cpp_source_path)
sys.path.append(cuda_source_path)
sys.path.append(opt_cuda_source_path)

# --- 2. Load Modules ---
modules = {}

try:
    import cpp_module
    modules['cpp'] = cpp_module
except ImportError: pass

try:
    import cuda_module
    modules['cuda'] = cuda_module
except ImportError: pass

try:
    import optimized_cuda_module
    modules['opt_cuda'] = optimized_cuda_module
except ImportError: pass

try:
    from wtconv.wtconv2d import WTConv2d
    print("[INFO] Dependencies loaded successfully.")
except ImportError as e:
    print(f"[ERROR] Could not import WTConv2d: {e}")
    sys.exit(1)

class RepoVerifier:
    def __init__(self, batch=2, channels=4, height=32, width=32, kernel_size=5, levels=1):
        self.B, self.C, self.H, self.W = batch, channels, height, width
        self.K, self.levels = kernel_size, levels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Kernel Parameters
        self.stride = 1
        self.pad = self.K // 2
        self.dwt_scale = 0.5
        self.idwt_scale = 0.5
        
        print(f"\n=== Configuration ===")
        print(f"  Input:   ({self.B}, {self.C}, {self.H}, {self.W})")
        print(f"  Kernel:  {self.K}x{self.K}")
        print(f"  Levels:  {self.levels}")
        print(f"  Device:  {self.device}")

        # Reference
        self.ref_layer = WTConv2d(self.C, self.C, kernel_size=self.K, wt_levels=self.levels, wt_type='db1')
        self.ref_layer.eval()
        self._isolate_core()
        
        self.x_torch = torch.randn(self.B, self.C, self.H, self.W)
        
        print("\n[Reference] Running PyTorch CPU Forward...")
        with torch.no_grad():
            self.out_ref = self.ref_layer(self.x_torch).numpy()
            
        self.weights = [self.ref_layer.wavelet_convs[i].weight.detach() for i in range(self.levels)]
        out_c, in_c, _, _ = self.weights[0].shape
        self.groups = self.C * 4 if in_c == 1 else 1
        print(f"  Groups:  {self.groups}")

    def _isolate_core(self):
        self.ref_layer.base_scale.weight.data.fill_(0.0)
        if self.ref_layer.base_scale.bias is not None: self.ref_layer.base_scale.bias.data.fill_(0.0)
        for s in self.ref_layer.wavelet_scale: s.weight.data.fill_(1.0)

    def verify(self):
        # 1. Verify C++
        self._run_test("C++ (CPU)", modules.get('cpp'), use_cuda=False)
        # 2. Verify Baseline CUDA
        self._run_test("Baseline CUDA", modules.get('cuda'), use_cuda=True)
        # 3. Verify Optimized CUDA
        self._run_test("Optimized CUDA", modules.get('opt_cuda'), use_cuda=True)

    def _run_test(self, name, mod, use_cuda):
        if mod is None:
            print(f"\n[SKIP] {name} module not found.")
            return

        print(f"\n=== Verifying {name} ===")
        try:
            if use_cuda:
                if not torch.cuda.is_available(): return
                x_in = self.x_torch.to(self.device)
                w_in = [w.to(self.device) for w in self.weights]
                out_t = mod.wtconv_forward(x_in, w_in, self.stride, self.pad, self.groups, self.dwt_scale, self.idwt_scale)
                out = out_t.cpu().numpy()
            else:
                x_in = self.x_torch.clone()
                w_in = [w.clone() for w in self.weights]
                out_t = mod.wtconv_forward(x_in, w_in, self.stride, self.pad, self.groups, self.dwt_scale, self.idwt_scale)
                out = out_t.numpy()

            diff = np.abs(self.out_ref - out).max()
            print(f"  -> Max Diff: {diff:.8f}")
            if diff < 1e-5: print(f"  ✅ {name} PASSED")
            else:           print(f"  ❌ {name} FAILED")
            
        except Exception as e:
            print(f"  [ERROR] Execution failed: {e}")

if __name__ == "__main__":
    RepoVerifier(levels=2).verify()