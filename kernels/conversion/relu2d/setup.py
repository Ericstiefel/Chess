from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='relu2d_cuda',
    ext_modules=[
        CUDAExtension(
            name='relu2d_cuda',
            sources=['relu2d_api.cpp', 'relu2d.cu'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
