// batch_norm_cuda.cpp

#include <torch/extension.h>
#include <vector>

// Helper macro for checking tensor properties for CUDA operations.
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)



void batch_norm_4d_cuda_launcher(const float* input, float* output,
                                 const float* gamma, const float* beta,
                                 int N, int C, int H, int W, float eps);

void batch_norm_4d_backward_cuda_launcher(const float* doutput, const float* input,
                                          float* dinput, const float* gamma,
                                          float* dgamma, float* dbeta,
                                          int N, int C, int H, int W, float eps);


torch::Tensor batch_norm_forward(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps) {

    // --- Input Validation ---
    CHECK_INPUT(input);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor (N, C, H, W)");
    TORCH_CHECK(gamma.dim() == 1 && beta.dim() == 1, "Gamma and Beta must be 1D tensors");

    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    TORCH_CHECK(gamma.size(0) == C && beta.size(0) == C, "Gamma and Beta must have size equal to the number of channels C");

    // --- Execution ---
    auto output = torch::empty_like(input);

    batch_norm_4d_cuda_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        N, C, H, W,
        eps
    );

    return output;
}


std::vector<torch::Tensor> batch_norm_backward(
    torch::Tensor doutput,
    torch::Tensor input,
    torch::Tensor gamma,
    float eps) {

    // --- Input Validation ---
    CHECK_INPUT(doutput);
    CHECK_INPUT(input);
    CHECK_INPUT(gamma);
    TORCH_CHECK(doutput.dim() == 4 && input.dim() == 4, "doutput and input must be 4D tensors");
    TORCH_CHECK(gamma.dim() == 1, "Gamma must be a 1D tensor");

    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    // --- Execution ---
    auto dinput = torch::empty_like(input);
    auto dgamma = torch::empty_like(gamma);
    auto dbeta = torch::empty_like(gamma);

    batch_norm_4d_backward_cuda_launcher(
        doutput.data_ptr<float>(),
        input.data_ptr<float>(),
        dinput.data_ptr<float>(),
        gamma.data_ptr<float>(),
        dgamma.data_ptr<float>(),
        dbeta.data_ptr<float>(),
        N, C, H, W,
        eps
    );

    return {dinput, dgamma, dbeta};
}


// --- Pybind11 Module Definition ---
// This macro creates the Python module and exposes the C++ functions.
// TORCH_EXTENSION_NAME is a macro that will be defined by the build system.

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &batch_norm_forward, "Batch Normalization Forward Pass (CUDA)");
    m.def("backward", &batch_norm_backward, "Batch Normalization Backward Pass (CUDA)");
}