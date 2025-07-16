#include <cuda_runtime.h>
#include <cmath>


__global__ void relu2d_forward_kernel(const float* input, float* output, int N, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * D) return;
    output[idx] = fmaxf(0.0f, input[idx]);
}

__global__ void relu2d_backward_kernel(const float* grad_output, const float* input, float* grad_input, int N, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * D) return;
    grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
}



extern "C" {

void relu2d_forward_launcher(const float* input, float* output, int N, int D) {
    int threads = 256;
    int blocks = (N * D + threads - 1) / threads;
    relu2d_forward_kernel<<<blocks, threads>>>(input, output, N, D);
}

void relu2d_backward_launcher(const float* grad_output, const float* input, float* grad_input, int N, int D) {
    int threads = 256;
    int blocks = (N * D + threads - 1) / threads;
    relu2d_backward_kernel<<<blocks, threads>>>(grad_output, input, grad_input, N, D);
}

}
