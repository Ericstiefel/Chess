#include <torch/extension.h>
#include <vector>

extern "C" {
void relu4d_forward_launcher(const float* input, float* output, int N, int C, int H, int W);
void relu4d_backward_launcher(const float* grad_output, const float* input, float* grad_input, int N, int C, int H, int W);
}

torch::Tensor relu4d_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor");

    auto output = torch::empty_like(input);
    
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    relu4d_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W
    );
    
    return output;
}

torch::Tensor relu4d_backward(torch::Tensor grad_output, torch::Tensor input) {
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(grad_output.dim() == 4, "grad_output must be a 4D tensor");
    TORCH_CHECK(input.dim() == 4, "input must be a 4D tensor");

    auto grad_input = torch::empty_like(input);

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    relu4d_backward_launcher(
        grad_output.data_ptr<float>(),
        input.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        N, C, H, W
    );

    return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &relu4d_forward, "ReLU4D forward (CUDA)");
    m.def("backward", &relu4d_backward, "ReLU4D backward (CUDA)");
}
