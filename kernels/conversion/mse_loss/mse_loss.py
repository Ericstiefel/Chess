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

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("CUDA not available, exiting.")
        exit()

    N = 1024
    y_hat_tensor = torch.rand(N, requires_grad=True, device=device)
    y_tensor = torch.rand(N, device=device)

    custom_loss_fn = MSELoss()
    custom_loss = custom_loss_fn(y_hat_tensor, y_tensor)
    
    custom_loss.backward()
    
    custom_grad = y_hat_tensor.grad.clone()
    
    y_hat_tensor.grad.zero_()

    pytorch_loss_fn = nn.MSELoss(reduction='mean')
    y_hat_pytorch = y_hat_tensor.detach().clone().requires_grad_(True)
    y_pytorch = y_tensor.detach().clone()
    
    pytorch_loss = pytorch_loss_fn(y_hat_pytorch, y_pytorch)
    pytorch_loss.backward()
    pytorch_grad = y_hat_pytorch.grad

    print(f"Custom CUDA MSE Loss: {custom_loss.item()}")
    print(f"PyTorch MSELoss:      {pytorch_loss.item()}")
    
    print("\nChecking loss difference...")
    if torch.allclose(custom_loss, pytorch_loss):
        print("Losses match!")
    else:
        print("Losses DO NOT match.")
        print(f"Difference: {torch.abs(custom_loss - pytorch_loss).item()}")

    print("\nChecking gradient difference...")
    if torch.allclose(custom_grad, pytorch_grad):
        print("Gradients match!")
    else:
        print("Gradients DO NOT match.")
        print(f"Max difference: {torch.max(torch.abs(custom_grad - pytorch_grad)).item()}")
