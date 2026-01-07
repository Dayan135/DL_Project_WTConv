from setuptools import setup, Extension
import pybind11
import sys

# Define compiler flags
extra_compile_args = []
extra_link_args = []

if sys.platform == "linux" or sys.platform == "linux2":
    extra_compile_args = ['-O3', '-fopenmp']
    extra_link_args = ['-fopenmp']
elif sys.platform == "darwin": # MacOS
    # MacOS usually needs specific OpenMP setup (libomp), handled via brew usually
    # Leaving empty to avoid build crash if libomp isn't found
    extra_compile_args = ['-O3']
    extra_link_args = []

ext_modules = [
    Extension(
        "cpp_module",
        ["pybind_module.cpp", "cpp_kernel.cpp"],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="cpp_module",
    version="0.1",
    ext_modules=ext_modules,
)