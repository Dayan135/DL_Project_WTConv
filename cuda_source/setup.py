import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Define compiler flags based on Operating System
if os.name == 'nt':  # Windows
    # Windows (MSVC) flags
    cxx_args = ['/O2', '/openmp']
    # NVCC flags (forwarded to host compiler via -Xcompiler)
    nvcc_args = ['-O3', '-Xcompiler', '/O2', '-Xcompiler', '/openmp']
else:  # Linux/Mac
    # GCC/Clang flags
    cxx_args = ['-O3', '-fopenmp', '-march=native']
    nvcc_args = ['-O3']

setup(
    name='cuda_module',
    ext_modules=[
        CUDAExtension('cuda_module', [
            'pybind_module.cpp',
            'cuda_kernel.cu',
<<<<<<< HEAD
        ]),
=======
        ],
        extra_compile_args={'cxx': cxx_args,
                            'nvcc': nvcc_args}
        )
>>>>>>> main
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)