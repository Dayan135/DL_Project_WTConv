import sys
import os
import torch
import numpy as np
from pathlib import Path

# --- 1. Dynamic Path Setup ---
# Handles Windows/Linux paths correctly using pathlib
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

# --- 2. Import Modules Safely ---
# Global flags to know what is available
HAS_CPP = False
HAS_CUDA = False

try:
    import cpp_module
    HAS_CPP = True
except ImportError:
    pass

try:
    import cuda_module
    HAS_CUDA = True
except ImportError:
    pass

try:
    from wtconv.wtconv2d import WTConv2d
    print("[INFO] Dependencies loaded successfully.")
except ImportError as e:
    print(f"[ERROR] Could not import WTConv2d: {e}")
    sys.exit(1)


class RepoVerifier:
    def __init__(self, batch=2, channels=4, height=32, width=32, kernel_size=5, levels=1):
        # 1. Architecture Config
        self.B, self.C, self.H, self.W = batch, channels, height, width
        self.K, self.levels = kernel_size, levels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 2. Kernel Parameters (Class Properties)
        self.stride = 1
        self.pad = self.K // 2
        # Validated: Arithmetic Mean (0.5) matches BGU's Haar implementation
        self.dwt_scale = 0.5
        self.idwt_scale = 0.5

        print(f"\n=== Configuration ===")
        print(f"  Input:   ({self.B}, {self.C}, {self.H}, {self.W})")
        print(f"  Kernel:  {self.K}x{self.K}")
        print(f"  Levels:  {self.levels}")
        print(f"  Scales:  DWT={self.dwt_scale}, IDWT={self.idwt_scale}")
        print(f"  Device:  {self.device}")

        # 3. Setup Reference Model
        self.ref_layer = WTConv2d(self.C, self.C, kernel_size=self.K, wt_levels=self.levels, wt_type='db1')
        self.ref_layer.eval()
        self._isolate_core_logic()
        
        # 4. Prepare Data
        self.x_torch = torch.randn(self.B, self.C, self.H, self.W)
        
        # 5. Run Reference (CPU)
        print("\n[Reference] Running PyTorch CPU Forward...")
        with torch.no_grad():
            self.out_ref = self.ref_layer(self.x_torch).numpy()
            
        # 6. Prepare Weights
        self.weights = [self.ref_layer.wavelet_convs[i].weight.detach() for i in range(self.levels)]
        
        # Infer Groups
        out_c, in_c, _, _ = self.weights[0].shape
        self.groups = self.C * 4 if in_c == 1 else 1
        print(f"  Groups:  {self.groups} ({'Depthwise' if self.groups > 1 else 'Dense'})")

    def _isolate_core_logic(self):
        """Disables residual connections and extra scaling to test the core algorithm."""
        self.ref_layer.base_scale.weight.data.fill_(0.0)
        if self.ref_layer.base_scale.bias is not None: self.ref_layer.base_scale.bias.data.fill_(0.0)
        for s in self.ref_layer.wavelet_scale:
            s.weight.data.fill_(1.0)
            if s.bias is not None: s.bias.data.fill_(0.0)

    def verify_cpp(self):
        if not HAS_CPP:
            print("\n[SKIP] C++ module not found.")
            return

        print("\n=== Verifying C++ Kernel (CPU) ===")
        
        # Prepare Inputs (Tensors on CPU)
        x_cpu = self.x_torch.clone() 
        w_cpu = [w.clone() for w in self.weights] 
        
        print(f"  -> Calling cpp_module.wtconv_forward")
        
        try:
            # The C++ module infers 'levels' from len(w_cpu)
            out_t = cpp_module.wtconv_forward(
                x_cpu, w_cpu, 
                self.stride, self.pad, self.groups, 
                self.dwt_scale, self.idwt_scale
            )
            out_cpp = out_t.numpy()
            self._check_result("C++ Kernel", out_cpp)
        except Exception as e:
            print(f"  [ERROR] C++ execution failed: {e}")

    def verify_cuda(self):
        if not HAS_CUDA or not torch.cuda.is_available():
            print("\n[SKIP] CUDA module not found or GPU unavailable.")
            return

        print("\n=== Verifying CUDA Kernel (GPU) ===")
        
        # Prepare Inputs (Tensors on GPU)
        x_c = self.x_torch.to(self.device)
        w_c = [w.to(self.device) for w in self.weights]
        
        print(f"  -> Calling cuda_module.wtconv_forward")

        try:
            # The CUDA module infers 'levels' from len(w_c)
            out_t = cuda_module.wtconv_forward(
                x_c, w_c, 
                self.stride, self.pad, self.groups, 
                self.dwt_scale, self.idwt_scale
            )
            out_cuda = out_t.cpu().numpy()
            self._check_result("CUDA Kernel", out_cuda)
        except Exception as e:
            print(f"  [ERROR] CUDA execution failed: {e}")

    def _check_result(self, name, output):
        diff = np.abs(self.out_ref - output).max()
        print(f"  -> Max Diff: {diff:.8f}")
        
        if diff < 1e-5:
            print(f"  ✅ {name} PASSED")
        else:
            print(f"  ❌ {name} FAILED")

def main():
    # You can easily change levels here to test scalability
    verifier = RepoVerifier(levels=2)
    
    # Run Tests
    verifier.verify_cpp()
    verifier.verify_cuda()

if __name__ == "__main__":
    main()