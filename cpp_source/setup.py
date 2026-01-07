from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

ext_modules = [
    Extension(
        "cpp_module",
        ["pybind_module.cpp", "cpp_kernel.cpp"],
        include_dirs=[pybind11.get_include()],
        language='c++'
    ),
]

setup(
    name="cpp_module",
    version="0.1",
    ext_modules=ext_modules,
)