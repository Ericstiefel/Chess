#include <torch/extension.h>
#include <vector>

// --- Helper Macros for Input Validation ---
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

extern "C" {
void conv2d_4d_forward_cuda_launcher(const float* input, float* output, const float* host_kernel,
                                     int B, int C, int H, int W);

void conv2d_4d_input_grad_cuda_launcher(const float* dOutput, float* dInput, const float* host_kernel,
                                        int B, int C, int H, int W);

void conv2d_4d_kernel_grad_cuda_launcher(const float* input, const float* dOutput, float* dKernel,
                                         int B, int C, int H, int W);
}


// --- C++ Wrapper for the Forward Pass ---
torch::Tensor conv_forward(
    torch::Tensor input,
    torch::Tensor kernel) {

    // Validate inputs
    CHECK_INPUT(input);
    CHECK_INPUT(kernel);
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor (B, C, H, W)");
    TORCH_CHECK(kernel.dim() == 2, "Kernel must be a 2D tensor (K, K)");
    TORCH_CHECK(kernel.size(0) == kernel.size(1), "Kernel must be square");
    // This implementation hard-codes a 3x3 kernel in the .cu file
    TORCH_CHECK(kernel.size(0) == 3, "This implementation only supports a 3x3 kernel");

    const int B = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    // Create an output tensor of the same shape as the input
    auto output = torch::empty_like(input);

    // Call the CUDA launcher
    conv2d_4d_forward_cuda_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        kernel.data_ptr<float>(),
        B, C, H, W
    );

    return output;
}

// --- C++ Wrapper for the Backward Pass ---
// This function computes both gradients (dInput and dKernel) needed for autograd.
std::vector<torch::Tensor> conv_backward(
    torch::Tensor dOutput,
    torch::Tensor input,
    torch::Tensor kernel) {

    // Validate inputs
    CHECK_INPUT(dOutput);
    CHECK_INPUT(input);
    CHECK_INPUT(kernel);

    const int B = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    // Create tensors to hold the gradients
    auto dInput = torch::empty_like(input);
    auto dKernel = torch::empty_like(kernel);

    // --- Compute Input Gradient ---
    conv2d_4d_input_grad_cuda_launcher(
        dOutput.data_ptr<float>(),
        dInput.data_ptr<float>(),
        kernel.data_ptr<float>(),
        B, C, H, W
    );

    // --- Compute Kernel Gradient ---
    conv2d_4d_kernel_grad_cuda_launcher(
        input.data_ptr<float>(),
        dOutput.data_ptr<float>(),
        dKernel.data_ptr<float>(),
        B, C, H, W
    );

    return {dInput, dKernel};
}


// --- Pybind11 Module Definition ---
// This creates the Python module `conv4d_cpp`, making the C++ functions
// available as `conv4d_cpp.forward(...)` and `conv4d_cpp.backward(...)`.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_forward, "4D Convolution Forward (CUDA)");
    m.def("backward", &conv_backward, "4D Convolution Backward (CUDA)");
}