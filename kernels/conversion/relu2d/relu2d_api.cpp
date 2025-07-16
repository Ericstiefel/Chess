
#include <torch/extension.h>
#include <vector>

extern "C" {
    void relu2d_forward_launcher(const float* input, float* output, int N, int D);
    void relu2d_backward_launcher(const float* grad_output, const float* input, float* grad_input, int N, int D);
}

// Forward wrapper
torch::Tensor relu2d_forward(torch::Tensor input) {
    TORCH_CHECK(input.dim() == 2, "Input must be a 2D tensor");
    auto output = torch::zeros_like(input);

    int N = input.size(0);
    int D = input.size(1);

    relu2d_forward_launcher(input.data_ptr<float>(), output.data_ptr<float>(), N, D);

    return output;
}

// Backward wrapper
torch::Tensor relu2d_backward(torch::Tensor grad_output, torch::Tensor input) {
    TORCH_CHECK(input.dim() == 2 && grad_output.dim() == 2, "Both tensors must be 2D");
    auto grad_input = torch::zeros_like(input);

    int N = input.size(0);
    int D = input.size(1);

    relu2d_backward_launcher(grad_output.data_ptr<float>(), input.data_ptr<float>(), grad_input.data_ptr<float>(), N, D);

    return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &relu2d_forward, "ReLU 2D forward (CUDA)");
    m.def("backward", &relu2d_backward, "ReLU 2D backward (CUDA)");
}
