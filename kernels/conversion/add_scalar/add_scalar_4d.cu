#include <cuda_runtime.h>

__global__ void add_scalar_4d_kernel(float* input, const int total_size, const float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_size) {
        input[idx] += value;
    }
}

void add_scalar_4d_cuda_launcher(float* input, int N, int C, int H, int W, float value) {
    int total_size = N * C * H * W;
    int threads_per_block = 256;
    int blocks_per_grid = (total_size + threads_per_block - 1) / threads_per_block;

    add_scalar_4d_kernel<<<blocks_per_grid, threads_per_block>>>(input, total_size, value);
}


void add_scalar_4d_cpu(float* input, int N, int C, int H, int W, float value) {
    // Calculate the total number of elements in the tensor.
    int total_size = N * C * H * W;

    for (int i = 0; i < total_size; ++i) {
        input[i] += value;
    }
}
