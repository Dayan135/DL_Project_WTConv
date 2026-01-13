import sys
import os
import pytest
import torch

# --- 1. Path Setup (Global) ---
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_DIR = os.path.join(HERE, "analysis")
REF_REPO_PATH = os.path.join(HERE, "Reference")

sys.path.append(ANALYSIS_DIR)
sys.path.insert(0, REF_REPO_PATH)

# Source paths for imports
sys.path.append(os.path.join(HERE, "cpp_source"))
sys.path.append(os.path.join(HERE, "cuda_source"))
sys.path.append(os.path.join(HERE, "optimized_cuda_source"))

# --- 2. Auto-Compilation ---
@pytest.fixture(scope="session", autouse=True)
def compile_extensions():
    """Compiles extensions once before any test runs."""
    import compile_utils
    print("\n[Pytest] Triggering Auto-Compilation...")
    compile_utils.compile_all()
    print("[Pytest] Compilation finished.\n")

# --- 3. Shared Fixtures ---
@pytest.fixture(scope="session")
def wtconv_class():
    try:
        from wtconv.wtconv2d import WTConv2d
        return WTConv2d
    except ImportError:
        pytest.fail("Could not import WTConv2d from Reference folder.")

@pytest.fixture
def reference_model(wtconv_class):
    """
    Returns a function that creates a standardized Reference Model 
    (bias disabled, scale=1.0) for comparison.
    """
    def _create(C, K, levels, device='cpu'):
        model = wtconv_class(C, C, kernel_size=K, wt_levels=levels, wt_type='db1').to(device)
        model.eval()
        
        # Isolate Core: Disable bias/scaling
        model.base_scale.weight.data.fill_(0.0)
        if model.base_scale.bias is not None:
            model.base_scale.bias.data.fill_(0.0)
        for s in model.wavelet_scale:
            s.weight.data.fill_(1.0)
            
        return model
    return _create

# Shared Test Parameters
@pytest.fixture(params=[
    (2, 4, 32, 32, 5, 1),
    (2, 4, 32, 32, 5, 2),
    (1, 8, 64, 64, 3, 1), 
    (1, 8, 64, 64, 7, 1),
], ids=["Batch2_L1", "Batch2_L2", "Large_K3", "Large_K7"])
def config(request):
    """Returns (B, C, H, W, K, levels)"""
    return request.param