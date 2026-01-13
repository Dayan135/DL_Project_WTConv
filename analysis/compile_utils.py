import os
import sys
import subprocess
import time
import shutil
from pathlib import Path

def setup_msvc_env():
    if shutil.which("cl"): return True
    print("[COMPILE] cl.exe not found. Attempting to locate VS DevCmd...")
    pf_x86 = os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")
    possible_paths = [
        f"{pf_x86}\\Microsoft Visual Studio\\2022\\Community\\Common7\\Tools\\VsDevCmd.bat",
        f"{pf_x86}\\Microsoft Visual Studio\\2022\\BuildTools\\Common7\\Tools\\VsDevCmd.bat",
        f"{pf_x86}\\Microsoft Visual Studio\\2019\\Community\\Common7\\Tools\\VsDevCmd.bat",
        f"{pf_x86}\\Microsoft Visual Studio\\2019\\BuildTools\\Common7\\Tools\\VsDevCmd.bat",
    ]
    vs_dev_cmd = next((p for p in possible_paths if os.path.exists(p)), None)
    if not vs_dev_cmd: return False

    cmd = f'"{vs_dev_cmd}" -arch=x64 && set'
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, text=True)
    if result.returncode != 0: return False
    
    for line in result.stdout.splitlines():
        if '=' in line:
            key, _, value = line.partition('=')
            os.environ[key] = value
    return True

def run_setup(source_dir):
    source_path = Path(source_dir).resolve()
    if not source_path.exists():
        print(f"[COMPILE] Error: Path not found: {source_path.name}")
        return False

    print(f"[COMPILE] Building: {source_path.name}...")
    original_cwd = os.getcwd()
    try:
        os.chdir(source_path)
        cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            print(f"[COMPILE] ❌ FAILED: {source_path.name}")
            print('\n'.join(result.stderr.splitlines()[-10:])) 
            return False
        else:
            print(f"[COMPILE] ✅ SUCCESS: {source_path.name}")
            return True
    except Exception as e:
        print(f"[COMPILE] ❌ Exception: {e}")
        return False
    finally:
        os.chdir(original_cwd)

def compile_all():
    if os.environ.get("SKIP_COMPILE", "0") == "1":
        print("[COMPILE] Skipped (SKIP_COMPILE=1)")
        return

    here = Path(__file__).parent
    root = here.parent
    
    if os.name == 'nt': setup_msvc_env()
    
    t0 = time.time()
    
    # 1. Baseline CPU
    run_setup(root / "cpp_source")
    
    # 2. Baseline CUDA
    run_setup(root / "cuda_source")

    # 3. Optimized CUDA (Fused V1)
    run_setup(root / "optimized_cuda_source")

    # 4. Optimized CUDA V2 (Fused Split) <--- ADDED
    run_setup(root / "optimized2_cuda_source")
        
    print(f"[COMPILE] Finished in {time.time() - t0:.2f}s\n")

if __name__ == "__main__":
    compile_all()