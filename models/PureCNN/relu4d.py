import torch
import torch.nn as nn
from torch.autograd import Function
import relu4d_cuda

class ReLU4DFunction(Function):
    @staticmethod
    def forward(ctx, input_tensor):
        ctx.save_for_backward(input_tensor)
        output = relu4d_cuda.forward(input_tensor)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.saved_tensors
        grad_input = relu4d_cuda.backward(grad_output, input_tensor)
        return grad_input

class ReLU4D(nn.Module):
    def forward(self, input_tensor):
        return ReLU4DFunction.apply(input_tensor)