#include <cuda_runtime.h>
#include <cmath>

__global__ void relu4d_forward_kernel(const float* input, float* output, int N, int C, int H, int W) {
    int w = threadIdx.x + blockIdx.x * blockDim.x;
    int h = threadIdx.y + blockIdx.y * blockDim.y;
    int nc = blockIdx.z;

    if (w >= W || h >= H) return;

    int n = nc / C;
    int c = nc % C;

    if (n >= N) return;

    int idx = ((n * C + c) * H + h) * W + w;
    output[idx] = fmaxf(0.0f, input[idx]);
}

__global__ void relu4d_backward_kernel(const float* grad_output, const float* input, float* grad_input, int N, int C, int H, int W) {
    int w = threadIdx.x + blockIdx.x * blockDim.x;
    int h = threadIdx.y + blockIdx.y * blockDim.y;
    int nc = blockIdx.z;

    if (w >= W || h >= H) return;
    
    int n = nc / C;
    int c = nc % C;

    if (n >= N) return;

    int idx = ((n * C + c) * H + h) * W + w;
    float gradient = (input[idx] > 0.0f) ? 1.0f : 0.0f;
    grad_input[idx] = grad_output[idx] * gradient;
}


extern "C" {
void relu4d_forward_launcher(const float* input, float* output, int N, int C, int H, int W) {
    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks((W + 7) / 8, (H + 7) / 8, N * C);
    relu4d_forward_kernel<<<numBlocks, threadsPerBlock>>>(input, output, N, C, H, W);
}

void relu4d_backward_launcher(const float* grad_output, const float* input, float* grad_input, int N, int C, int H, int W) {
    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks((W + 7) / 8, (H + 7) / 8, N * C);
    relu4d_backward_kernel<<<numBlocks, threadsPerBlock>>>(grad_output, input, grad_input, N, C, H, W);
}
}