import torch
import os
from torch.utils.cpp_extension import load

# This compiles the extension inline. Alternatively, use setup.py for pip install.
module_path = os.path.dirname(__file__)
add_scalar_4d_cuda = load(
    name="add_scalar_4d_cuda",
    sources=[
        os.path.join(module_path, "add_scalar_4d.cpp"),
        os.path.join(module_path, "add_scalar_4d.cu"),
    ],
    verbose=True,
)

def add_scalar_4d(input_tensor: torch.Tensor, value: float) -> torch.Tensor:
    if not input_tensor.is_contiguous():
        input_tensor = input_tensor.contiguous()
    add_scalar_4d_cuda.add_scalar_4d(input_tensor, value)
    return input_tensor
