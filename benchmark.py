import sys
import os
import time
import torch
import numpy as np

# 1. Setup Paths
# Add 'Reference' to path to import the original WTConv
sys.path.append(os.path.join(os.path.dirname(__file__), 'Reference'))
# Add 'cpp_source' to path to import your compiled C++ module
sys.path.append(os.path.join(os.path.dirname(__file__), 'cpp_source'))

try:
    from wtconv import WTConv2d
    print("[INFO] Successfully imported WTConv2d from Reference.")
except ImportError:
    print("[ERROR] Could not import WTConv2d. Check if 'Reference' folder exists and is the BGU repo.")
    sys.exit(1)

try:
    import cpp_module
    print("[INFO] Successfully imported cpp_module.")
except ImportError:
    print("[ERROR] Could not import cpp_module. Did you run 'python setup.py build_ext --inplace' in cpp_source?")
    sys.exit(1)

# ==========================================
# Configuration
# ==========================================
BATCH_SIZE = 4
IN_CHANNELS = 16
OUT_CHANNELS = 16
HEIGHT = 64
WIDTH = 64
KERNEL_SIZE = 5
WT_LEVELS = 1  # YOUR C++ CODE SUPPORTS ONLY 1 LEVEL
DEVICE = 'cpu' # C++ kernel is CPU-only, so we compare on CPU

def run_benchmark():
    print(f"\n[SETUP] Batch={BATCH_SIZE}, In={IN_CHANNELS}, Out={OUT_CHANNELS}, H={HEIGHT}, W={WIDTH}, K={KERNEL_SIZE}")
    
    # -------------------------------------------------
    # 1. Initialize PyTorch Model (Reference)
    # -------------------------------------------------
    # We force wt_levels=1 because your C++ kernel is hardcoded for 1 level logic.
    ref_model = WTConv2d(IN_CHANNELS, OUT_CHANNELS, kernel_size=KERNEL_SIZE, wt_levels=WT_LEVELS).to(DEVICE)
    
    # Important: WTConv2d usually uses a 'depthwise' or grouped convolution internally in the wavelet domain.
    # Your C++ code implements a DENSE convolution (groups=1).
    # To make them mathematically identical, we must ensure the Reference model 
    # is using the same logic, or we accept that we are benchmarking different math operations.
    # 
    # *Assumption*: We act as if we extracted the weights from the reference and treated them as dense
    # for the C++ kernel.
    
    # Create Random Input
    input_tensor = torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH, device=DEVICE, requires_grad=True)
    
    # Extract Weights for C++
    # The BGU WTConv2d usually has a member `self.wt_conv` which is the Conv2d layer.
    # We need to detach it and convert to numpy.
    if hasattr(ref_model, 'wt_conv'):
        weight_tensor = ref_model.wt_conv.weight.detach().cpu()
        print(f"[INFO] Extracted weights with shape: {weight_tensor.shape}")
    else:
        print("[ERROR] Could not find 'wt_conv' in reference model.")
        return

    # Prepare Numpy arrays for C++
    input_np = input_tensor.detach().cpu().numpy()
    weight_np = weight_tensor.numpy()
    
    # -------------------------------------------------
    # 2. Correctness Check: FORWARD
    # -------------------------------------------------
    print("\n--- Correctness Check: Forward ---")
    
    # Reference Forward
    ref_output = ref_model(input_tensor)
    
    # C++ Forward
    # Note: Reference might use padding differently. 
    # Standard WTConv usually does padding=kernel//2 inside.
    padding = KERNEL_SIZE // 2 
    stride = 1
    
    cpp_output = cpp_module.wtconv_forward(input_np, weight_np, stride, padding)
    
    # Convert ref to numpy for comparison
    ref_output_np = ref_output.detach().cpu().numpy()
    
    # Compare
    # Tolerance is slightly loose due to float precision differences
    if np.allclose(ref_output_np, cpp_output, atol=1e-4):
        print("[PASS] Forward pass results match!")
    else:
        diff = np.abs(ref_output_np - cpp_output).max()
        print(f"[FAIL] Forward pass mismatch. Max diff: {diff}")
        # Continue anyway to check speed

    # -------------------------------------------------
    # 3. Correctness Check: BACKWARD
    # -------------------------------------------------
    print("\n--- Correctness Check: Backward ---")
    
    # Reference Backward
    grad_output = torch.randn_like(ref_output)
    ref_model.zero_grad()
    ref_output.backward(grad_output)
    ref_grad_input = input_tensor.grad.detach().cpu().numpy()
    ref_grad_weight = ref_model.wt_conv.weight.grad.detach().cpu().numpy()
    
    # C++ Backward
    grad_output_np = grad_output.detach().cpu().numpy()
    cpp_grad_input, cpp_grad_weight = cpp_module.wtconv_backward(grad_output_np, input_np, weight_np, stride, padding)
    
    if np.allclose(ref_grad_input, cpp_grad_input, atol=1e-4):
        print("[PASS] Backward pass (Input Grads) match!")
    else:
        print(f"[FAIL] Backward pass (Input Grads) mismatch. Max diff: {np.abs(ref_grad_input - cpp_grad_input).max()}")

    if np.allclose(ref_grad_weight, cpp_grad_weight, atol=1e-4):
        print("[PASS] Backward pass (Weight Grads) match!")
    else:
        print(f"[FAIL] Backward pass (Weight Grads) mismatch. Max diff: {np.abs(ref_grad_weight - cpp_grad_weight).max()}")

    # -------------------------------------------------
    # 4. Performance Benchmark
    # -------------------------------------------------
    print("\n--- Performance Benchmark (Forward Pass) ---")
    iterations = 50
    
    # Measure PyTorch
    start_time = time.time()
    for _ in range(iterations):
        _ = ref_model(input_tensor)
    torch_time = (time.time() - start_time) / iterations
    
    # Measure C++
    start_time = time.time()
    for _ in range(iterations):
        _ = cpp_module.wtconv_forward(input_np, weight_np, stride, padding)
    cpp_time = (time.time() - start_time) / iterations
    
    print(f"PyTorch Average Time: {torch_time*1000:.4f} ms")
    print(f"C++     Average Time: {cpp_time*1000:.4f} ms")
    
    if cpp_time < torch_time:
        print(f"\n>> C++ is {torch_time/cpp_time:.2f}x FASTER")
    else:
        print(f"\n>> C++ is {cpp_time/torch_time:.2f}x SLOWER")

if __name__ == "__main__":
    run_benchmark()