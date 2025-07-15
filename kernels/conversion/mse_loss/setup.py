from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mse_loss_cuda',
    ext_modules=[
        CUDAExtension('mse_loss_cuda', [
            'mse_loss_api.cpp',
            'mse_loss.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
