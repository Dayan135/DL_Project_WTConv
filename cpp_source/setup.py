import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# Compiler Flags
if os.name == 'nt':  # Windows
    extra_compile_args = ['/O2', '/openmp']
else:  # Linux
    extra_compile_args = ['-O3', '-fopenmp', '-march=native']

setup(
    name='cpp_module',
    ext_modules=[
        CppExtension('cpp_module', [
            'pybind_module.cpp',
            'cpp_kernel.cpp',
        ],
        extra_compile_args=extra_compile_args
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)