#include <torch/extension.h>
#include <vector>

// CUDA declarations
void add_scalar_4d_cuda_launcher(float* input, int N, int C, int H, int W, float value);

// CPU fallback (optional)
void add_scalar_4d_cpu(float* input, int N, int C, int H, int W, float value);

void add_scalar_4d(torch::Tensor input, float value) {
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    float* input_ptr = input.data_ptr<float>();

    add_scalar_4d_cuda_launcher(input_ptr, N, C, H, W, value);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_scalar_4d", &add_scalar_4d, "Add scalar to 4D tensor (CUDA)");
}
