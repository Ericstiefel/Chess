from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name="add_scalar_4d",
    ext_modules=[
        CUDAExtension(
            name="add_scalar_4d",
            sources=["add_scalar_4d.cpp", "add_scalar_4d.cu"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
