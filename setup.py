from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

ext_modules = [
    Pybind11Extension(
        "cpp_module",
        ["cpp_source/pybind_module.cpp", "cpp_source/cpp_kernel.cpp"],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=["-std=c++11"],
    )
]

setup(
    name="cpp_module",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
