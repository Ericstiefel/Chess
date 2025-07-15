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

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("CUDA not available, exiting.")
        exit()

    rows, cols = 16, 1024
    input_tensor = torch.randn(rows, cols, requires_grad=True, device=device)

    # Custom Softmax
    custom_softmax = Softmax()
    custom_output = custom_softmax(input_tensor)
    custom_output.sum().backward()
    custom_grad = input_tensor.grad.clone()
    
    # Reset grad
    input_tensor.grad.zero_()

    # PyTorch Softmax
    pytorch_softmax = nn.Softmax(dim=1)
    pytorch_output = pytorch_softmax(input_tensor)
    pytorch_output.sum().backward()
    pytorch_grad = input_tensor.grad

    print("Checking forward pass difference...")
    if torch.allclose(custom_output, pytorch_output):
        print("Forward outputs match!")
    else:
        print("Forward outputs DO NOT match.")
        print(f"Max difference: {torch.max(torch.abs(custom_output - pytorch_output)).item()}")

    print("\nChecking backward pass difference...")
    if torch.allclose(custom_grad, pytorch_grad, atol=1e-6):
        print("Gradients match!")
    else:
        print("Gradients DO NOT match.")
        print(f"Max difference: {torch.max(torch.abs(custom_grad - pytorch_grad)).item()}")
