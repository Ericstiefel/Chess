import torch
import my_extension.batch_norm_cuda as _C

class BatchNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma, beta, eps=1e-5):
        output = _C.forward(input, gamma, beta, eps)
        ctx.save_for_backward(input, gamma, beta, output)
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, gamma, beta, output = ctx.saved_tensors
        eps = ctx.eps
        dinput, dgamma, dbeta = _C.backward(grad_output, input, gamma, eps)
        return dinput, dgamma, dbeta, None  # Last is for eps (no grad)
