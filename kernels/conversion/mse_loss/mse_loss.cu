#include <cuda_runtime.h>
#include <cmath>

__global__ void mse_loss_forward_kernel(const float* y_hat, const float* y, float* loss, int N) {
    __shared__ float partial_sums[256];
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx = threadIdx.x;

    if (global_idx < N) {
        float diff = y_hat[global_idx] - y[global_idx];
        partial_sums[local_idx] = diff * diff;
    } else {
        partial_sums[local_idx] = 0.0f;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (local_idx < stride) {
            partial_sums[local_idx] += partial_sums[local_idx + stride];
        }
        __syncthreads();
    }

    if (local_idx == 0) {
        loss[blockIdx.x] = partial_sums[0];
    }
}

__global__ void mse_loss_backward_kernel(const float* y_hat, const float* y, float* dL_dy_hat, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dL_dy_hat[idx] = 2.0f * (y_hat[idx] - y[idx]) / N;
    }
}

extern "C" {
void mse_loss_forward_launcher(const float* y_hat, const float* y, float* loss, int N) {
    dim3 threadsPerBlock(256);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x);
    mse_loss_forward_kernel<<<numBlocks, threadsPerBlock>>>(y_hat, y, loss, N);
}

void mse_loss_backward_launcher(const float* y_hat, const float* y, float* dL_dy_hat, int N) {
    dim3 threadsPerBlock(256);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x);
    mse_loss_backward_kernel<<<numBlocks, threadsPerBlock>>>(y_hat, y, dL_dy_hat, N);
}
}
