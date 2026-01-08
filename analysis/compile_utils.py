import os
import sys
import subprocess
import time
from pathlib import Path

def run_setup(source_dir):
    """
    Runs 'python setup.py build_ext --inplace' in the specified directory.
    """
    source_path = Path(source_dir).resolve()
    if not source_path.exists():
        print(f"[COMPILE] Error: Source path not found: {source_path}")
        return False

    print(f"[COMPILE] Building extension in: {source_path.name}...")
    
    # Store current dir to restore later
    original_cwd = os.getcwd()
    
    try:
        os.chdir(source_path)
        # Run setup.py build_ext --inplace
        # We use sys.executable to ensure we use the same Python interpreter
        cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
        
        # Capture output to keep console clean, print only on error
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            check=False
        )
        
        if result.returncode != 0:
            print(f"[COMPILE] ❌ Build FAILED for {source_path.name}")
            print(result.stderr)
            return False
        else:
            print(f"[COMPILE] ✅ Build SUCCESS for {source_path.name}")
            return True
            
    except Exception as e:
        print(f"[COMPILE] ❌ Exception during build: {e}")
        return False
    finally:
        os.chdir(original_cwd)

def compile_all():
    """
    Compiles both C++ and CUDA extensions.
    """
    # Assuming this script is in DL_PROJECT/analysis/
    # And sources are in DL_PROJECT/cpp_source/ and DL_PROJECT/cuda_source/
    
    here = Path(__file__).parent
    root = here.parent
    
    t0 = time.time()
    
    # 1. Compile C++ Module
    cpp_src = root / "cpp_source"
    if not run_setup(cpp_src):
        print("[COMPILE] ⚠️ Warning: cpp_module build failed. Scripts may crash if it's missing.")

    # 2. Compile CUDA Module
    cuda_src = root / "cuda_source"
    if not run_setup(cuda_src):
        print("[COMPILE] ⚠️ Warning: cuda_module build failed. Scripts may crash if it's missing.")
        
    print(f"[COMPILE] Finished in {time.time() - t0:.2f}s\n")

if __name__ == "__main__":
    compile_all()