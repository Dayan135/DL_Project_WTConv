from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_module',
    ext_modules=[
        CUDAExtension('cuda_module', [
            'pybind_module.cpp',
            'cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)