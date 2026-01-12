import sys
import os
import pytest
import torch

# --- 1. Path Setup (Global) ---
# We do this at the top level so tests can import modules easily
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_DIR = os.path.join(HERE, "analysis")
REF_REPO_PATH = os.path.join(HERE, "Reference")

# Add analysis to path to find compile_utils
sys.path.append(ANALYSIS_DIR)
# Add Reference to path to find wtconv
sys.path.insert(0, REF_REPO_PATH)

# Add Source paths so we can import the built extensions
sys.path.append(os.path.join(HERE, "cpp_source"))
sys.path.append(os.path.join(HERE, "cuda_source"))
sys.path.append(os.path.join(HERE, "optimized_cuda_source"))

# --- 2. Session-Scoped Compilation ---
@pytest.fixture(scope="session", autouse=True)
def compile_extensions():
    """
    Automatically compiles C++/CUDA extensions once at the start of the test session.
    """
    import compile_utils
    print("\n[Pytest] Triggering Auto-Compilation...")
    compile_utils.compile_all()
    print("[Pytest] Compilation finished.\n")

# --- 3. Shared Fixtures ---
@pytest.fixture(scope="session")
def wtconv_class():
    """Returns the Reference WTConv2d class."""
    try:
        from wtconv.wtconv2d import WTConv2d
        return WTConv2d
    except ImportError:
        pytest.fail("Could not import WTConv2d from Reference folder.")

@pytest.fixture
def isolate_core():
    """Helper to strip bias/scaling from a model for pure kernel testing."""
    def _isolate(model, levels):
        # Disable base convolution bias/scale
        model.base_scale.weight.data.fill_(0.0)
        if model.base_scale.bias is not None:
            model.base_scale.bias.data.fill_(0.0)
        
        # Set wavelet scales to 1.0 (identity)
        for s in model.wavelet_scale:
            s.weight.data.fill_(1.0)
        return model
    return _isolate