from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='relu4d_cuda',
    ext_modules=[
        CUDAExtension('relu4d_cuda', [
            'relu4d_api.cpp',
            'relu4d.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
