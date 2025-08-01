import torch
import torch.nn as nn
from torch.autograd import Function
import cross_entropy_loss_cuda

class CrossEntropyLossFunction(Function):
    @staticmethod
    def forward(ctx, y_hat, y):
        ctx.save_for_backward(y_hat, y)
        loss = cross_entropy_loss_cuda.forward(y_hat, y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y_hat, y = ctx.saved_tensors
        grad_input = cross_entropy_loss_cuda.backward(y_hat, y)
        return grad_input * grad_output, None

class CrossEntropyLoss(nn.Module):
    def forward(self, y_hat, y):
        return CrossEntropyLossFunction.apply(y_hat, y)