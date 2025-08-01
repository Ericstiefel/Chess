import torch
import torch.nn as nn
from torch.autograd import Function
import softmax_cuda

class SoftmaxFunction(Function):
    @staticmethod
    def forward(ctx, input_tensor):
        output = softmax_cuda.forward(input_tensor)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        grad_input = softmax_cuda.backward(grad_output.contiguous(), output)
        return grad_input

class Softmax(nn.Module):
    def forward(self, input_tensor):
        return SoftmaxFunction.apply(input_tensor)

