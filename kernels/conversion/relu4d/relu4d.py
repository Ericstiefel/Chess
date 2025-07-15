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

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("CUDA not available, exiting.")
        exit()

    N, C, H, W = 4, 3, 16, 16
    input_tensor = torch.randn(N, C, H, W, requires_grad=True, device=device)

    # Custom ReLU
    custom_relu = ReLU4D()
    custom_output = custom_relu(input_tensor)
    custom_output.sum().backward()
    custom_grad = input_tensor.grad.clone()
    
    # Reset grad
    input_tensor.grad.zero_()

    # PyTorch ReLU
    pytorch_relu = nn.ReLU()
    pytorch_output = pytorch_relu(input_tensor)
    pytorch_output.sum().backward()
    pytorch_grad = input_tensor.grad

    print("Checking forward pass difference...")
    if torch.allclose(custom_output, pytorch_output):
        print("Forward outputs match!")
    else:
        print("Forward outputs DO NOT match.")
        print(f"Max difference: {torch.max(torch.abs(custom_output - pytorch_output)).item()}")

    print("\nChecking backward pass difference...")
    if torch.allclose(custom_grad, pytorch_grad):
        print("Gradients match!")
    else:
        print("Gradients DO NOT match.")
        print(f"Max difference: {torch.max(torch.abs(custom_grad - pytorch_grad)).item()}")
