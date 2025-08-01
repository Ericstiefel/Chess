from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='softmax_cuda',
    ext_modules=[
        CUDAExtension('softmax_cuda', [
            'softmax_api.cpp',
            'softmax.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
