#include <torch/extension.h>
#include <vector>

extern "C" {
void softmax_forward_launcher(const float* input, float* output, int rows, int cols);
void softmax_backward_launcher(const float* grad_output, const float* output, float* grad_input, int rows, int cols);
}

torch::Tensor softmax_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dim() == 2, "Input must be a 2D tensor");

    auto output = torch::empty_like(input);
    
    int rows = input.size(0);
    int cols = input.size(1);

    softmax_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows, cols
    );
    
    return output;
}

torch::Tensor softmax_backward(torch::Tensor grad_output, torch::Tensor output) {
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(grad_output.dim() == 2, "grad_output must be a 2D tensor");
    TORCH_CHECK(output.dim() == 2, "output must be a 2D tensor");

    auto grad_input = torch::empty_like(output);

    int rows = output.size(0);
    int cols = output.size(1);

    softmax_backward_launcher(
        grad_output.data_ptr<float>(),
        output.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        rows, cols
    );

    return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softmax_forward, "Softmax forward (CUDA)");
    m.def("backward", &softmax_backward, "Softmax backward (CUDA)");
}
