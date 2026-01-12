import pytest
import torch
import numpy as np

# Import your compiled extensions
# We wrap them in try-except so pytest collects tests even if compilation fails (though they will error at runtime)
try: import cpp_module
except ImportError: cpp_module = None

try: import cuda_module
except ImportError: cuda_module = None

try: import optimized_cuda_module
except ImportError: optimized_cuda_module = None

# --- Test Parameters ---
PARAMS = [
    # (batch, channels, height, width, kernel, levels)
    (2, 4, 32, 32, 5, 1),
    (2, 4, 32, 32, 5, 2),
    (1, 8, 64, 64, 3, 1), # Test kernel 3
    (1, 8, 64, 64, 7, 1), # Test kernel 7
]

@pytest.mark.parametrize("B, C, H, W, K, levels", PARAMS)
class TestWTConvForward:
    
    @pytest.fixture(autouse=True)
    def setup(self, B, C, H, W, K, levels, wtconv_class, isolate_core):
        """Prepares reference output and inputs once per test case."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Prepare Reference Model
        ref_layer = wtconv_class(C, C, kernel_size=K, wt_levels=levels, wt_type='db1')
        ref_layer.eval()
        isolate_core(ref_layer, levels)
        
        # 2. Prepare Inputs
        self.x_torch = torch.randn(B, C, H, W)
        
        # 3. Run Reference (CPU)
        with torch.no_grad():
            self.out_ref = ref_layer(self.x_torch).numpy()
            
        # 4. Extract Weights for Custom Modules
        self.weights = [ref_layer.wavelet_convs[i].weight.detach() for i in range(levels)]
        
        # 5. Common Params
        self.stride = 1
        self.pad = K // 2
        self.dwt_scale = 0.5
        self.idwt_scale = 0.5
        out_c, in_c, _, _ = self.weights[0].shape
        self.groups = C * 4 if in_c == 1 else 1

    # --- TESTS ---

    def test_cpp_forward(self):
        if cpp_module is None: pytest.skip("cpp_module not installed")
        
        # C++ module runs on CPU
        x_in = self.x_torch.clone()
        w_in = [w.clone() for w in self.weights]
        
        out_t = cpp_module.wtconv_forward(
            x_in, w_in, self.stride, self.pad, self.groups, 
            self.dwt_scale, self.idwt_scale
        )
        
        diff = np.abs(self.out_ref - out_t.numpy()).max()
        assert diff < 1e-5, f"CPP Forward failed. Max diff: {diff}"

    def test_baseline_cuda_forward(self):
        if cuda_module is None: pytest.skip("cuda_module not installed")
        if not torch.cuda.is_available(): pytest.skip("No CUDA device")

        x_in = self.x_torch.to(self.device)
        w_in = [w.to(self.device) for w in self.weights]
        
        out_t = cuda_module.wtconv_forward(
            x_in, w_in, self.stride, self.pad, self.groups, 
            self.dwt_scale, self.idwt_scale
        )
        
        diff = np.abs(self.out_ref - out_t.cpu().numpy()).max()
        assert diff < 1e-5, f"Baseline CUDA Forward failed. Max diff: {diff}"

    def test_optimized_cuda_forward(self):
        if optimized_cuda_module is None: pytest.skip("opt_cuda_module not installed")
        if not torch.cuda.is_available(): pytest.skip("No CUDA device")

        x_in = self.x_torch.to(self.device)
        w_in = [w.to(self.device) for w in self.weights]
        
        out_t = optimized_cuda_module.wtconv_forward(
            x_in, w_in, self.stride, self.pad, self.groups, 
            self.dwt_scale, self.idwt_scale
        )
        
        diff = np.abs(self.out_ref - out_t.cpu().numpy()).max()
        assert diff < 1e-5, f"Optimized CUDA Forward failed. Max diff: {diff}"
        