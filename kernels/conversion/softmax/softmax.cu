#include <cuda_runtime.h>
#include <cmath>
#include <limits>

__global__ void softmax_forward_kernel(const float* __restrict__ input, float* __restrict__ output, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float sdata[];

    // Step 1: Find the maximum value in the row (Parallel Reduction)
    float max_val = -std::numeric_limits<float>::infinity();
    for (int j = tid; j < cols; j += blockDim.x) {
        max_val = fmaxf(max_val, input[row * cols + j]);
    }
    sdata[tid] = max_val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    max_val = sdata[0];

    // Step 2: Calculate the sum of exponents (Parallel Reduction)
    float sum_val = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        sum_val += expf(input[row * cols + j] - max_val);
    }
    sdata[tid] = sum_val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    sum_val = sdata[0];

    // Step 3: Compute the final softmax values
    for (int j = tid; j < cols; j += blockDim.x) {
        output[row * cols + j] = expf(input[row * cols + j] - max_val) / sum_val;
    }
}

__global__ void softmax_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ output,
    float* __restrict__ grad_input,
    int rows, int cols
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float sdata[];

    // Step 1: Compute dot(grad_output, output) for the row
    float dot_val = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        dot_val += grad_output[row * cols + j] * output[row * cols + j];
    }
    sdata[tid] = dot_val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    dot_val = sdata[0];

    // Step 2: Compute final gradient: grad_input = output * (grad_output - dot_val)
    for (int j = tid; j < cols; j += blockDim.x) {
        float s = output[row * cols + j];
        float dy = grad_output[row * cols + j];
        grad_input[row * cols + j] = s * (dy - dot_val);
    }
}

extern "C" {
void softmax_forward_launcher(const float* input, float* output, int rows, int cols) {
    int threads = 256;
    int blocks = rows;
    size_t sharedMem = threads * sizeof(float);
    softmax_forward_kernel<<<blocks, threads, sharedMem>>>(input, output, rows, cols);
}

void softmax_backward_launcher(const float* grad_output, const float* output, float* grad_input, int rows, int cols) {
    int threads = 256;
    int blocks = rows;
    size_t sharedMem = threads * sizeof(float);
    softmax_backward_kernel<<<blocks, threads, sharedMem>>>(grad_output, output, grad_input, rows, cols);
}
}