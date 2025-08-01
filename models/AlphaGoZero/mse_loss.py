import torch
import torch.nn as nn
from torch.autograd import Function
import mse_loss_cuda

class MSELossFunction(Function):
    @staticmethod
    def forward(ctx, y_hat, y):
        ctx.save_for_backward(y_hat, y)
        loss = mse_loss_cuda.forward(y_hat, y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y_hat, y = ctx.saved_tensors
        grad_input = mse_loss_cuda.backward(y_hat, y)
        return grad_input * grad_output, None

class MSELoss(nn.Module):
    def forward(self, y_hat, y):
        return MSELossFunction.apply(y_hat, y)

