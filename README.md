# High-Performance Wavelet Convolution (WTConv)

This repository contains a highly optimized implementation of the **Wavelet Convolution (WTConv)** layer, accelerating the original PyTorch implementation using custom **C++** and **CUDA** kernels.

The project moves the computationally heavy operations from standard PyTorch layers to custom JIT-compiled extensions that utilize **Kernel Fusion**, **Shared Memory tiling**, and **Parallel Reduction**, achieving significant speedups over the reference implementation.

Based on the article "Wavelet Convolutions for Large Receptive Fields" by Shahaf E. Finder, Roy Amoyal, Eran Treister, Oren Freifeld: 
https://arxiv.org/abs/2407.05848

And on BGU-CS-VIL/WTConv git repo as reference to speedup: 
https://github.com/BGU-CS-VIL/WTConv

## 🚀 Features

* **Reference Implementation:** A pure PyTorch implementation (based on `BGU-CS-VIL/WTConv`) for correctness verification.
* **C++ (CPU) Extension:** OpenMP-parallelized implementation for efficient CPU execution.
* **Baseline CUDA:** A direct port of the algorithm to CUDA.
* **Optimized CUDA (Fused):** A highly optimized kernel that fuses the Discrete Wavelet Transform (DWT), Convolution, and Inverse DWT (IDWT) into single kernels to minimize global memory access.
    * **Forward Pass:** ~8.7x faster than PyTorch.
    * **Backward Pass:** ~4.5x faster than PyTorch.
* **Optimized V2 (Experimental):** An alternative approach.

## 📂 Repository Structure

```text
DL_Project_WTConv/
├── analysis/                 # Benchmarking and compilation scripts
│   ├── benchmark_backward.py          # Run speed comparisons
│   └── compile_utils.py      # JIT compilation utility
├── cpp_source/               # C++ (CPU) Source Code
├── cuda_source/              # Baseline CUDA Source Code
├── optimized_cuda_source/    # Optimized (Fused) CUDA Source Code
├── optimized2_cuda_source/   # Experimental "Pre-calc" CUDA Source Code
├── Reference/                # Original PyTorch WTConv implementation
├── tests/                    # PyTest suite for correctness verification
├── experiment_results/       # Logs and plots from benchmarks
├── experiment.py             # Main experiment runner
├── inference.py              # Inference scripts
├── model_surgery.py          # Utils to replace Conv2d with WTConv in models
├── train.py                  # Training script
├── requirements.txt          # Python dependencies
└── README.md
```

## 🛠️ Setup & Installation

- **Prerequisites**

    - **OS**: Windows or Linux

    - **GPU**: NVIDIA GPU with CUDA support (Compute Capability 6.0+)

    - **Compiler**: MSVC (Windows) or GCC (Linux) available in your system path.


- **Environment Setup (Optional)(Recommended: Conda)**

    ```Bash
    # Create a new environment
    conda create -n wt_conv python=3.10
    conda activate wt_conv
    ```

- **Install Dependencies**

    Install the required Python packages for testing, benchmarking, and building.

    ```Bash
    pip install -r requirements.txt
    ```
## ⚙️ Compilation

This project uses a custom JIT (Just-In-Time) compilation script to build the C++ and CUDA extensions.

Run the compilation utility:

```Bash
python analysis/compile_utils.py
```

- If successful, you will see `✅ SUCCESS` messages for `cpp_source`, `cuda_source`, `optimized_cuda_source` and `optimized2_cuda_source`.

- **Windows Note**: If you encounter an "Access Denied" error, ensure no Python processes (like a running Jupyter kernel or `pytest`) are locking the `.pyd` files. Close them and try again.

## Running Tests

We use `pytest` to ensure that the custom C++ and CUDA implementations produce identical outputs and identical gradients to the PyTorch reference.

```Bash
pytest tests/
```
Test Breakdown:

- `test_cpp.py`: Validates the C++ (CPU) Forward and Backward logic.

- `test_cuda.py`: Validates the Baseline CUDA implementation.

- `test_opts.py`: Validates the Optimized CUDA implementations (1 and 2) against PyTorch Autograd.

## 📊 Benchmarking & Analysis

To analyze the performance improvements, run the benchmark script `analysis/benchmark_backward.py`. This script automatically compiles the extensions (unless skipped) and compares the runtime of the Forward and Backward passes across all implementations.

- **Basic usage:**

    ```Bash
    python analysis/benchmark_backward.py
    ```

- **Customizing the Benchmark**

    You can customize the input dimensions, batch size, and kernel parameters using CLI flags:

    ```Bash
    python analysis/benchmark_backward.py --batch 32 --channels 128 --kernel 5 --iters 100
    ```

    **Available Flags:**


    | Flag        | Default | Description                                                     |
    |-------------|---------|-----------------------------------------------------------------|
    | `--batch`   | 16      | Batch size (N)                                                  |
    | `--channels`| 64      | Input channels (C)                                              |
    | `--height`  | 64      | Input height (H)                                                |
    | `--width`   | 64      | Input width (W)                                                 |
    | `--kernel`  | 5       | Convolution kernel size (K)                                     |
    | `--levels`  | 1       | Wavelet decomposition levels                                    |
    | `--iters`   | 50      | Number of benchmark iterations for timing average               |
    | `--warmup`  | 30      | Number of warmup iterations before timing                       |
    | `--no-build`| False   | Skip auto-compilation step (useful if already compiled)         |


## Reproducing Experiments
We provide a fully automated experiment suite run_experiment_full.py that handles training, inference benchmarking, cleanup, and data analysis in a single step.

- **Run the Full Suite (Paper Reproduction)**
To replicate the exact experiments reported in our work (Training ResNet18 for 6 epochs on 3 different seeds, across all wavelet levels), run:
   ```Bash
    python run_experiment_full.py --epochs 6 --seeds 10 2 367 --levels 1 2 3 4
    ```


This script will automatically:

   - Train the model using 4 different implementations (reference, cuda, cuda_opt, cuda_opt2) across all specified levels and seeds.
     
   - Benchmark Inference on the trained checkpoints to measure latency and throughput.
     
   - Clean Up heavy checkpoint files to save disk space (optional).
     
   - Analyze Results and generate the plots and tables found in `--full_analysis_output/`

- **Custom Experiments:**
   You can customize the benchmark using CLI flags.

   - Example: Quick Sanity Check (1 Epoch, Level 1 only):
       ```Bash
          python run_experiment_full.py --epochs 1 --levels 1 --impls cuda_opt2
       ```

 | Flag               | Default  | Description                                                     |
 |--------------------|----------|-----------------------------------------------------------------|
 | `--epochs`         | 6        | Number of training epochs per run.                              |
 | `--batch-size`     | 64       | Batch size for training and inference.                          |
 | `--model`          | resnet18 | Model architecture to use (resnet18 or resnet50).               |
 | `--seeds`          | 10 2 367 | Space-separated list of random seeds for multiple runs.         |
 | `--levels`         | 1 2 3 4  | Space-separated list of Wavelet decomposition levels to test.   |
 | `--impls`          | all      | Specific implementations to run (e.g. cuda_opt2).               |
 | `--device`         | cuda     | Compute device (cuda or cpu).                                   |
 | `--skip-train`     | False    | Skip the training phase (useful for re-running inference).      |
 | `--skip-inference` | False    | Skip the inference benchmark phase.                             |
 | `--no-cleanup`     | False    | Keep .pth checkpoints after the run (Uses disk space).          |
 | `--no-report-clean`| False    | Do not delete old report folders before starting.               |


- **Output Data**
   - After the run completes, check the `--full_analysis_output/` directory for:
   
   - training_summary_by_level.csv: Detailed training metrics (Forward/Backward ms, VRAM).
   
   - inference_summary_by_level.csv: Inference latency and accuracy stats.
   
   - *.png: Visualizations including the Scalability Chart, Speedup Heatmap, and Kernel Breakdown.






