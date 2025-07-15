from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="batch_norm_cuda",
    ext_modules=[
        CUDAExtension(
            name="batch_norm_cuda",
            sources=['batch_norm_api.cpp', 'batch_norm_cuda.cu'],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
