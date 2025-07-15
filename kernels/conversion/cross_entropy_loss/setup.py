from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cross_entropy_loss_cuda',
    ext_modules=[
        CUDAExtension('cross_entropy_loss_cuda', [
            'cross_entropy_loss_api.cpp',
            'cross_entropy_loss.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
