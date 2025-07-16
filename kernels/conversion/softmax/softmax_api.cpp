#include <torch/extension.h>
#include <vector>

extern "C" {
void softmax_forward_launcher(const float* input, float* output, int size);
void softmax_backward_launcher(const float* grad_output, const float* output, float* grad_input, int size);
}

torch::Tensor softmax_forward(torch::Tensor input) {
    // Check for 1D tensor
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dim() == 1, "Input must be a 1D tensor");

    auto output = torch::empty_like(input);
    
    // Get the size of the 1D tensor
    int size = input.size(0);

    // Call the 1D launcher
    softmax_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    
    return output;
}

torch::Tensor softmax_backward(torch::Tensor grad_output, torch::Tensor output) {
    // Check for 1D tensors
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(grad_output.dim() == 1, "grad_output must be a 1D tensor");
    TORCH_CHECK(output.dim() == 1, "output must be a 1D tensor");

    auto grad_input = torch::empty_like(output);

    // Get the size of the 1D tensor
    int size = output.size(0);

    // Call the 1D launcher
    softmax_backward_launcher(
        grad_output.data_ptr<float>(),
        output.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        size
    );

    return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softmax_forward, "Softmax forward (CUDA 1D)");
    m.def("backward", &softmax_backward, "Softmax backward (CUDA 1D)");
}