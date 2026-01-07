# WTConv C++ Kernel Implementation

This project implements a custom C++ kernel for Wavelet Convolutions (WTConv), designed to replicate the logic of the official [WTConv repository](https://github.com/BGU-CS-VIL/WTConv).

The kernel implements:
1. **Haar Wavelet Transform (DWT)**
2. **Convolution in Wavelet Domain** (supports Dense and Depthwise/Grouped)
3. **Inverse Haar Wavelet Transform (IDWT)**

## 📂 Project Structure

```text
.
├── cpp_source/             # C++ Source Code
│   ├── cpp_kernel.cpp      # Kernel Implementation
│   ├── cpp_kernel.h        # Header
│   ├── pybind_module.cpp   # Python Bindings (PyBind11)
│   └── setup.py            # Build Script
├── Reference/              # Cloned BGU-CS-VIL/WTConv repository
├── benchmark.py            # Performance comparison
├── verify_equivalence.py   # Basic logic verification
├── verify_against_repo.py  # Verification against official BGU weights
├── requirements.txt        # Python dependencies
└── README.md