import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Define compiler flags
if os.name == 'nt':  # Windows
    cxx_args = ['/O2', '/openmp']
    nvcc_args = ['-O3', '-Xcompiler', '/O2', '-Xcompiler', '/openmp']
else:  # Linux
    cxx_args = ['-O3', '-fopenmp']
    nvcc_args = ['-O3']

setup(
    name='optimized2_cuda_module',  # <--- 1. IMPORT NAME IN PYTHON
    ext_modules=[
        CUDAExtension(
            name='optimized2_cuda_module', # <--- 2. EXTENSION NAME
            sources=[
                'pybind_module.cpp',
                'cuda_kernel.cu',
            ],
            extra_compile_args={
                'cxx': cxx_args,
                'nvcc': nvcc_args
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)