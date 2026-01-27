
import subprocess
import time
import sys
import os
import glob
import argparse
import shutil  # <--- Added for directory cleanup

# --- Constants (Paths) ---
TRAIN_SCRIPT = os.path.join("full_model_tests", "train.py")
INFERENCE_SCRIPT = os.path.join("full_model_tests", "inference.py")
ANALYSIS_SCRIPT = "analyze_full_results.py" 
CACHE_DIR = "model_cache"

# Output directories to clean before starting
REPORT_DIRS = [
    "inference_reports", 
    "training_reports", 
    "full_analysis_output",
]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Full End-to-End WTConv Benchmark Suite")
    
    # Core Training Config
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs per run")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet50"], help="Model architecture")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    # Variations
    parser.add_argument("--seeds", type=int, nargs="+", default=[10, 2, 367], 
                        help="List of random seeds to run")
    
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="List of wavelet levels to test")
    
    parser.add_argument("--impls", type=str, nargs="+", 
                        default=["reference", "cuda", "cuda_opt", "cuda_opt2"],
                        choices=["reference", "cuda", "cuda_opt", "cuda_opt2"],
                        help="List of implementations to test")

    # Workflow Flags
    parser.add_argument("--skip-train", action="store_true", help="Skip training phase")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference phase")
    parser.add_argument("--no-cleanup", action="store_true", help="Do not delete checkpoints after run")
    parser.add_argument("--no-report-clean", action="store_true", help="Do not delete old reports at start")

    return parser.parse_args()

def clean_old_reports():
    print(f"\n==================================================")
    print(f"🧹 PRE-RUN CLEANUP (Removing old reports)")
    print(f"==================================================")
    for folder in REPORT_DIRS:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                print(f"   🗑️  Wiped: {folder}/")
            except Exception as e:
                print(f"   ⚠️  Could not remove {folder}: {e}")
        
        # Re-create empty folders immediately so scripts don't crash
        os.makedirs(folder, exist_ok=True)
        print(f"   ✨ Created empty: {folder}/")
    print("")

def main():
    args = parse_arguments()

    # --- 0. PRE-RUN CLEANUP ---
    if not args.no_report_clean:
        clean_old_reports()
    else:
        print("⏩ Skipping Report Cleanup (--no-report-clean set)")

    print(f"==================================================")
    print(f"🚀 STARTING AUTOMATED SUITE (Train -> Inference -> Cleanup -> Analysis)")
    print(f"   Implementations: {args.impls}")
    print(f"   Seeds:           {args.seeds}")
    print(f"   Epochs: {args.epochs} | Batch: {args.batch_size} | Levels: {args.levels}")
    print(f"==================================================\n")

    failed_runs = []

    # --- PART 1: TRAINING LOOP ---
    if not args.skip_train:
        for seed in args.seeds:
            for level in args.levels:
                for impl in args.impls:
                    print(f"\n▶️  [TRAIN] {impl} | Level {level} | Seed {seed}")
                    cmd = [
                        sys.executable, TRAIN_SCRIPT,
                        "--impl", impl,
                        "--model", args.model,
                        "--epochs", str(args.epochs),
                        "--batch-size", str(args.batch_size),
                        "--wt-levels", str(level),
                        "--device", args.device,
                        "--save-weights",
                        "--seed", str(seed)
                    ]

                    try:
                        subprocess.run(cmd, check=True) 
                    except subprocess.CalledProcessError as e:
                        print(f"   ❌ Failed: {impl} (Seed {seed})")
                        print(f"      Exit Code: {e.returncode}")
                        failed_runs.append(f"{impl}_L{level}_S{seed}")
                    
                    # Short cooldown
                    time.sleep(1) 
    else:
        print("⏩ Skipping Training (--skip-train set)")

    # --- PART 2: INFERENCE ---
    if args.skip_inference:
        print("\n⏩ Skipping Inference (--skip-inference set)")
    else:
        print(f"\n==================================================")
        print(f"▶️  STARTING INFERENCE BENCHMARK")
        print(f"==================================================\n")

        ckpts = glob.glob(os.path.join(CACHE_DIR, "*.pth"))
        if not ckpts:
            print("❌ No checkpoints found. Cannot run inference.")
        else:
            print(f"🔍 Starting Inference on {len(ckpts)} checkpoints...")
            try:
                subprocess.run([
                    sys.executable, INFERENCE_SCRIPT,
                    "--model", args.model, 
                    "--batch-size", str(args.batch_size),
                    "--device", args.device,
                    "--cache-dir", CACHE_DIR
                ], check=True)
            except subprocess.CalledProcessError:
                print("❌ Inference script failed.")

    # --- PART 3: CHECKPOINT CLEANUP ---
    if args.no_cleanup:
        print("\n⏩ Skipping Checkpoint Cleanup (--no-cleanup set).")
    else:
        print(f"\n==================================================")
        print(f"🧹 CHECKPOINT CLEANUP")
        print(f"==================================================\n")

        abs_cache_dir = os.path.abspath(CACHE_DIR)
        files_to_delete = glob.glob(os.path.join(abs_cache_dir, "*.pth"))

        if not files_to_delete:
            print(f"    ⚠️  No checkpoint files found.")
        else:
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                    print(f"    🗑️  Deleted: {os.path.basename(file_path)}")
                except OSError as e:
                    print(f"    ❌  Error deleting {os.path.basename(file_path)}: {e}")

    # --- PART 4: ANALYSIS ---
    print(f"\n==================================================")
    print(f"📊 STARTING ANALYSIS")
    print(f"==================================================\n")

    if not os.path.exists(ANALYSIS_SCRIPT):
        print(f"❌ Error: Analysis script not found at {ANALYSIS_SCRIPT}")
    else:
        try:
            subprocess.run([sys.executable, ANALYSIS_SCRIPT], check=True)
            print("\n✅  Analysis Completed Successfully.")
        except subprocess.CalledProcessError as e:
            print(f"\n❌  Analysis FAILED with exit code {e.returncode}.")

    print("\n==================================================")
    print("🏁  FULL SUITE COMPLETED")
    print("==================================================")

if __name__ == "__main__":
    if not os.path.exists(TRAIN_SCRIPT):
        print(f"Error: Missing {TRAIN_SCRIPT}")
        sys.exit(1)
    if not os.path.exists(INFERENCE_SCRIPT):
        print(f"Error: Missing {INFERENCE_SCRIPT}")
        sys.exit(1)
        
    main()