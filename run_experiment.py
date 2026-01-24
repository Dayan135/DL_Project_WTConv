
# import subprocess
# import time
# import sys
# import os
# import glob

# # --- Configuration ---
# # The list of implementations you want to train
# IMPLEMENTATIONS = [
#     "reference",
#     "cuda",
#     "cuda_opt",
#     "cuda_opt2"
# ]

# # Shared training settings
# MODEL = "resnet18"
# EPOCHS = 3
# BATCH_SIZE = 64
# WT_LEVELS = 3
# DEVICE = "cuda"

# # Paths (Relative to project root)
# TRAIN_SCRIPT = os.path.join("full_model_tests", "train.py")
# INFERENCE_SCRIPT = os.path.join("full_model_tests", "inference.py")
# CACHE_DIR = "model_cache"  # Where checkpoints are saved (must match train.py defaults)

# def main():
#     print(f"==================================================")
#     print(f"🚀 STARTING AUTOMATED SUITE (Train -> Inference -> Cleanup)")
#     print(f"   Implementations: {IMPLEMENTATIONS}")
#     print(f"   Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | Levels: {WT_LEVELS}")
#     print(f"==================================================\n")

#     failed_runs = []

#     # --- PART 1: TRAINING LOOP ---
#     for impl in IMPLEMENTATIONS:
#         print(f"--------------------------------------------------")
#         print(f"▶️  [TRAIN] Starting: [{impl}]")
#         print(f"--------------------------------------------------")

#         cmd = [
#             sys.executable, TRAIN_SCRIPT,
#             "--impl", impl,
#             "--model", MODEL,
#             "--epochs", str(EPOCHS),
#             "--batch-size", str(BATCH_SIZE),
#             "--wt-levels", str(WT_LEVELS),
#             "--device", DEVICE,
#             "--save-weights"
#         ]

#         try:
#             subprocess.run(cmd, check=True)
#             print(f"\n✅  [{impl}] Training Completed.")
#         except subprocess.CalledProcessError as e:
#             print(f"\n❌  [{impl}] FAILED with exit code {e.returncode}.")
#             failed_runs.append(impl)
#         except KeyboardInterrupt:
#             print("\n⚠️  Suite interrupted. Stopping.")
#             sys.exit(1)

#         print("    Waiting 3 seconds for GPU cooldown...")
#         time.sleep(3)

#     # --- PART 2: INFERENCE ---
#     if len(failed_runs) == len(IMPLEMENTATIONS):
#         print("\n❌  All training runs failed. Skipping inference.")
#         sys.exit(1)

#     print(f"\n==================================================")
#     print(f"▶️  STARTING INFERENCE BENCHMARK")
#     print(f"==================================================\n")

#     inference_cmd = [
#         sys.executable, INFERENCE_SCRIPT,
#         "--model", MODEL,
#         "--batch-size", str(BATCH_SIZE),
#         "--device", DEVICE,
#         # Ensure inference looks in the correct project-root cache folder
#         "--cache-dir", CACHE_DIR 
#     ]

#     try:
#         subprocess.run(inference_cmd, check=True)
#         print("\n✅  Inference Benchmark Completed.")
#     except subprocess.CalledProcessError as e:
#         print(f"\n❌  Inference FAILED with exit code {e.returncode}.")
#         # We generally don't want to delete weights if inference failed, 
#         # so you might want to exit here.
#         sys.exit(1)

#     # --- PART 3: CLEANUP ---
#     print(f"\n==================================================")
#     print(f"🧹 STARTING CLEANUP (Deleting Checkpoints)")
#     print(f"==================================================\n")

#     # Find all .pth files in the cache directory
#     # Note: We use the absolute path to be safe
#     abs_cache_dir = os.path.abspath(CACHE_DIR)
#     files_to_delete = glob.glob(os.path.join(abs_cache_dir, "*.pth"))

#     if not files_to_delete:
#         print(f"    ⚠️  No checkpoint files found in {abs_cache_dir}")
#     else:
#         for file_path in files_to_delete:
#             try:
#                 os.remove(file_path)
#                 print(f"    🗑️  Deleted: {os.path.basename(file_path)}")
#             except OSError as e:
#                 print(f"    ❌  Error deleting {os.path.basename(file_path)}: {e}")

#     print("\n==================================================")
#     print("🏁  FULL SUITE COMPLETED")
#     print("==================================================")

# if __name__ == "__main__":
#     # Sanity checks
#     if not os.path.exists(TRAIN_SCRIPT):
#         print(f"Error: Missing {TRAIN_SCRIPT}")
#         sys.exit(1)
#     if not os.path.exists(INFERENCE_SCRIPT):
#         print(f"Error: Missing {INFERENCE_SCRIPT}")
#         sys.exit(1)
        
#     main()


import subprocess
import time
import sys
import os
import glob

# --- Configuration ---
# The list of implementations you want to train
IMPLEMENTATIONS = [
    "reference",
    "cuda",
    "cuda_opt",
    "cuda_opt2"
]

# Shared training settings
MODEL = "resnet18"
EPOCHS = 3
BATCH_SIZE = 64
WT_LEVELS = 2
DEVICE = "cuda"
SEEDS = [42 , 100 , 2024]  # <--- NEW: List of seeds for multiple runs

# Paths (Relative to project root)
TRAIN_SCRIPT = os.path.join("full_model_tests", "train.py")
INFERENCE_SCRIPT = os.path.join("full_model_tests", "inference.py")
ANALYSIS_SCRIPT = os.path.join("full_model_tests", "analyze_results.py" ) # <--- NEW: Path to analysis script
CACHE_DIR = "model_cache"

def main():
    print(f"==================================================")
    print(f"🚀 STARTING AUTOMATED SUITE (Train -> Inference -> Cleanup -> Analysis)")
    print(f"   Implementations: {IMPLEMENTATIONS}")
    print(f"   Seeds: {SEEDS}")
    print(f"   Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | Levels: {WT_LEVELS}")
    print(f"==================================================\n")

    failed_runs = []

    # # --- PART 1: TRAINING LOOP ---
    # for impl in IMPLEMENTATIONS:
    #     print(f"--------------------------------------------------")
    #     print(f"▶️  [TRAIN] Starting: [{impl}]")
    #     print(f"--------------------------------------------------")

    #     cmd = [
    #         sys.executable, TRAIN_SCRIPT,
    #         "--impl", impl,
    #         "--model", MODEL,
    #         "--epochs", str(EPOCHS),
    #         "--batch-size", str(BATCH_SIZE),
    #         "--wt-levels", str(WT_LEVELS),
    #         "--device", DEVICE,
    #         "--save-weights"
    #     ]

    #     try:
    #         subprocess.run(cmd, check=True)
    #         print(f"\n✅  [{impl}] Training Completed.")
    #     except subprocess.CalledProcessError as e:
    #         print(f"\n❌  [{impl}] FAILED with exit code {e.returncode}.")
    #         failed_runs.append(impl)
    #     except KeyboardInterrupt:
    #         print("\n⚠️  Suite interrupted. Stopping.")
    #         sys.exit(1)

    #     print("    Waiting 3 seconds for GPU cooldown...")
    #     time.sleep(3)
    # --- PART 1: TRAINING LOOP (Nested) ---
    for seed in SEEDS:
        print(f"\n🌱 === Starting Batch for SEED {seed} ===")
        
        for impl in IMPLEMENTATIONS:
            print(f"   ▶️  [TRAIN] {impl} (Seed {seed})")

            cmd = [
                sys.executable, TRAIN_SCRIPT,
                "--impl", impl,
                "--model", MODEL,
                "--epochs", str(EPOCHS),
                "--batch-size", str(BATCH_SIZE),
                "--wt-levels", str(WT_LEVELS),
                "--device", DEVICE,
                "--save-weights",
                "--seed", str(seed)  # <--- Passing the seed
            ]

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                print(f"   ❌ Failed: {impl} (Seed {seed})")
                # We continue to the next one even if this fails
            
            # Short cooldown
            time.sleep(2)    

    # --- PART 2: INFERENCE ---
    if len(failed_runs) == len(IMPLEMENTATIONS):
        print("\n❌  All training runs failed. Skipping inference.")
        sys.exit(1)

    print(f"\n==================================================")
    print(f"▶️  STARTING INFERENCE BENCHMARK")
    print(f"==================================================\n")

    # inference_cmd = [
    #     sys.executable, INFERENCE_SCRIPT,
    #     "--model", MODEL,
    #     "--batch-size", str(BATCH_SIZE),
    #     "--device", DEVICE,
    #     "--cache-dir", CACHE_DIR 
    # ]

    # try:
    #     subprocess.run(inference_cmd, check=True)
    #     print("\n✅  Inference Benchmark Completed.")
    # except subprocess.CalledProcessError as e:
    #     print(f"\n❌  Inference FAILED with exit code {e.returncode}.")
    #     sys.exit(1)


    print(f"\n🔍 Starting Inference on ALL checkpoints...")
    subprocess.run([
        sys.executable, INFERENCE_SCRIPT,
        "--model", MODEL, 
        "--batch-size", str(BATCH_SIZE),
        "--device", DEVICE,
        "--cache-dir", CACHE_DIR
    ], check=True)    

    # --- PART 3: CLEANUP ---
    print(f"\n==================================================")
    print(f"🧹 STARTING CLEANUP (Deleting Checkpoints)")
    print(f"==================================================\n")

    abs_cache_dir = os.path.abspath(CACHE_DIR)
    files_to_delete = glob.glob(os.path.join(abs_cache_dir, "*.pth"))

    if not files_to_delete:
        print(f"    ⚠️  No checkpoint files found in {abs_cache_dir}")
    else:
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                print(f"    🗑️  Deleted: {os.path.basename(file_path)}")
            except OSError as e:
                print(f"    ❌  Error deleting {os.path.basename(file_path)}: {e}")

    # --- PART 4: ANALYSIS ---
    print(f"\n==================================================")
    print(f"📊 STARTING ANALYSIS (Generating Plots & Tables)")
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