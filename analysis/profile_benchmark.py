import os
import sys
import torch
import time
import pandas as pd

# --- Setup ---
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import compile_utils

print("--- Triggering Auto-Compilation ---")
compile_utils.compile_all()

ref_path = os.path.join(HERE, "Reference")
cuda_path = os.path.join(HERE, "cuda_source")
opt_cuda_path = os.path.join(HERE, "optimized_cuda_source")

sys.path.insert(0, ref_path)
sys.path.append(cuda_path)
sys.path.append(opt_cuda_path)

try: import cuda_module
except: pass
try: import optimized_cuda_module
except: pass

def benchmark_op(name, fn, iters=100):
    # Warmup
    for _ in range(10): fn()
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / iters

def run_profile(batch=16, channels=64, height=64, width=64, kernel_size=5):
    print(f"\n=== Profiling: {batch}x{channels}x{height}x{width} (K={kernel_size}) ===")
    
    device = torch.device("cuda")
    input_t = torch.randn(batch, channels, height, width, device=device)
    
    # 1. DWT Setup
    # DWT Output is 4x Channels, Half Res
    dwt_out_shape = (batch, channels*4, height//2, width//2)
    dwt_out_sim = torch.randn(*dwt_out_shape, device=device)
    
    # 2. Conv Setup
    # Weights for Depthwise: [Out, In/G, K, K]. 
    # Here Groups=Channels*4. So In/G = 1.
    weight_t = torch.randn(channels*4, 1, kernel_size, kernel_size, device=device)
    
    stride = 1
    pad = kernel_size // 2
    
    results = []

    # --- Profile Baseline ---
    if 'cuda_module' in sys.modules:
        mod = cuda_module
        tag = "Baseline"
        
        t_dwt = benchmark_op("DWT", lambda: mod.dwt_forward(input_t, 0.5))
        t_conv = benchmark_op("Conv", lambda: mod.conv_forward(dwt_out_sim, weight_t, stride, pad))
        # IDWT input is same shape as DWT output
        t_idwt = benchmark_op("IDWT", lambda: mod.idwt_forward(dwt_out_sim, 0.5))
        
        total = t_dwt + t_conv + t_idwt
        results.append({
            "Module": tag, 
            "DWT (ms)": t_dwt, 
            "Conv (ms)": t_conv, 
            "IDWT (ms)": t_idwt, 
            "Total (sum)": total
        })

    # --- Profile Optimized ---
    if 'optimized_cuda_module' in sys.modules:
        mod = optimized_cuda_module
        tag = "Optimized"
        
        t_dwt = benchmark_op("DWT", lambda: mod.dwt_forward(input_t, 0.5))
        t_conv = benchmark_op("Conv", lambda: mod.conv_forward(dwt_out_sim, weight_t, stride, pad))
        t_idwt = benchmark_op("IDWT", lambda: mod.idwt_forward(dwt_out_sim, 0.5))
        
        total = t_dwt + t_conv + t_idwt
        results.append({
            "Module": tag, 
            "DWT (ms)": t_dwt, 
            "Conv (ms)": t_conv, 
            "IDWT (ms)": t_idwt, 
            "Total (sum)": total
        })

    df = pd.DataFrame(results)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df)
    
    # Calculate Ratios
    if not df.empty:
        conv_ratio = df.iloc[0]["Conv (ms)"] / df.iloc[0]["Total (sum)"]
        print(f"\n[Analysis] Convolution takes {conv_ratio:.1%} of the runtime.")

if __name__ == "__main__":
    # Small Case
    run_profile(batch=16, channels=64, height=64, width=64, kernel_size=5)
    
    # Large Case
    run_profile(batch=64, channels=5, height=224, width=224, kernel_size=7)