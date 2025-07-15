import torch
import torch.nn as nn
from torch.autograd import Function

# Import the compiled C++/CUDA extension
import conv4d_cpp

class Conv4DFunction(Function):


    @staticmethod
    def forward(ctx, input_tensor, kernel_tensor):

        input_tensor = input_tensor.cuda().contiguous()
        kernel_tensor = kernel_tensor.cuda().contiguous()

        ctx.save_for_backward(input_tensor, kernel_tensor)
        output = conv4d_cpp.forward(input_tensor, kernel_tensor)
        return output

    @staticmethod
    def backward(ctx, grad_output):

        grad_output = grad_output.cuda().contiguous()

        # Retrieve tensors saved in the forward pass
        input_tensor, kernel_tensor = ctx.saved_tensors

        # Call the C++ 'backward' function from our compiled extension
        grad_input, grad_kernel = conv4d_cpp.backward(grad_output, input_tensor, kernel_tensor)

        return grad_input, grad_kernel


class Conv4D(nn.Module):

    def __init__(self):
        super(Conv4D, self).__init__()
        kernel_initial = torch.randn(3, 3, dtype=torch.float32)
        self.kernel = nn.Parameter(kernel_initial)

    def forward(self, input_tensor):

        return Conv4DFunction.apply(input_tensor, self.kernel)

