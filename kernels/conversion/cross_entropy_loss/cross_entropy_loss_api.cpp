#include <torch/extension.h>
#include <vector>

extern "C" {
void cross_entropy_loss_forward_launcher(const float* y_hat, const float* y, float* loss, int N);
void cross_entropy_loss_backward_launcher(const float* y_hat, const float* y, float* dL_dy_hat, int N);
}

torch::Tensor cross_entropy_loss_forward(torch::Tensor y_hat, torch::Tensor y) {
    TORCH_CHECK(y_hat.is_cuda(), "y_hat must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(y_hat.is_contiguous(), "y_hat must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
    TORCH_CHECK(y_hat.sizes() == y.sizes(), "y_hat and y must have the same shape");

    int N = y_hat.numel();
    int num_blocks = (N + 255) / 256;
    
    auto options = torch::TensorOptions().device(y_hat.device()).dtype(y_hat.dtype());
    torch::Tensor block_losses = torch::zeros({num_blocks}, options);

    cross_entropy_loss_forward_launcher(
        y_hat.data_ptr<float>(),
        y.data_ptr<float>(),
        block_losses.data_ptr<float>(),
        N
    );
    
    return block_losses.sum();
}

torch::Tensor cross_entropy_loss_backward(torch::Tensor y_hat, torch::Tensor y) {
    TORCH_CHECK(y_hat.is_cuda(), "y_hat must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(y_hat.is_contiguous(), "y_hat must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
    TORCH_CHECK(y_hat.sizes() == y.sizes(), "y_hat and y must have the same shape");

    auto dL_dy_hat = torch::empty_like(y_hat);
    int N = y_hat.numel();

    cross_entropy_loss_backward_launcher(
        y_hat.data_ptr<float>(),
        y.data_ptr<float>(),
        dL_dy_hat.data_ptr<float>(),
        N
    );

    return dL_dy_hat;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cross_entropy_loss_forward, "Cross Entropy Loss forward (CUDA)");
    m.def("backward", &cross_entropy_loss_backward, "Cross Entropy Loss backward (CUDA)");
}