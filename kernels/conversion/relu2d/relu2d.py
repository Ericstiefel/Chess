import torch
import relu2d_cuda  # This is the compiled module name

class ReLU2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = relu2d_cuda.forward(input)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = relu2d_cuda.backward(grad_output, input)
        return grad_input

def relu2d(input):
    return ReLU2DFunction.apply(input)
