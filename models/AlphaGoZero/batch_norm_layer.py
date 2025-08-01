import torch
import batch_norm_cuda as _C

class BatchNormCuda(torch.nn.Module):

    def __init__(self, num_features, eps=1e-5):
        super(BatchNormCuda, self).__init__()
        if not isinstance(num_features, int):
            raise TypeError("num_features must be an integer.")
            
        self.num_features = num_features
        self.eps = eps
        
        self.gamma = torch.nn.Parameter(torch.ones(num_features))
        self.beta = torch.nn.Parameter(torch.zeros(num_features))

    def forward(self, input):
        if input.device != self.gamma.device:
            input = input.to(self.gamma.device)
            
        return BatchNormFunction.apply(input, self.gamma, self.beta, self.eps)

class BatchNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma, beta, eps=1e-5):
        input_cuda = input.contiguous().cuda()
        gamma_cuda = gamma.contiguous().cuda()
        beta_cuda = beta.contiguous().cuda()

        output = _C.forward(input_cuda, gamma_cuda, beta_cuda, eps)
        
        ctx.save_for_backward(input_cuda, gamma_cuda, output) 
        ctx.eps = eps
        
        return output

    @staticmethod
    def backward(ctx, grad_output):

        grad_output_cuda = grad_output.contiguous().cuda()
        
        input, gamma, output = ctx.saved_tensors
        eps = ctx.eps
        
        # Call the compiled CUDA backward function
        dinput, dgamma, dbeta = _C.backward(grad_output_cuda, input, gamma, eps)
        
        return dinput, dgamma, dbeta, None