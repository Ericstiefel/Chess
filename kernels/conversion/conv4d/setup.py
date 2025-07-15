from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='conv4d_cpp',
    ext_modules=[
        CUDAExtension(
            # This name must match the one in PYBIND11_MODULE
            name='conv4d_cpp',
            sources=[
                'conv4d_api.cpp',
                'conv4d.cu',
            ],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
