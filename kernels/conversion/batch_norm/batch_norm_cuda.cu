#include <cuda_runtime.h>
#include <math.h>

#define TILE_DIM 64


template <int TILE>
__global__ void batch_norm_4d(const float* __restrict__ input,
                              float* __restrict__ output,
                              const float* __restrict__ gamma,
                              const float* __restrict__ beta,
                              int N, int C, int H, int W,
                              float eps) {
    int c = blockIdx.x;
    if (c >= C) return;

    __shared__ float local_sum[TILE];
    __shared__ float local_sqsum[TILE];

    int tid = threadIdx.x;
    int stride = blockDim.x;

    float sum = 0.0f;
    float sqsum = 0.0f;

    // Step 1: Compute per-channel mean and variance
    for (int idx = tid; idx < N * H * W; idx += stride) {
        int n = idx / (H * W);
        int hw = idx % (H * W);
        int h = hw / W;
        int w = hw % W;

        float val = input[((n * C + c) * H + h) * W + w];
        sum += val;
        sqsum += val * val;
    }

    local_sum[tid] = sum;
    local_sqsum[tid] = sqsum;

    __syncthreads();

    for (int s = TILE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            local_sum[tid] += local_sum[tid + s];
            local_sqsum[tid] += local_sqsum[tid + s];
        }
        __syncthreads();
    }

    float mean = local_sum[0] / (N * H * W);
    float var  = local_sqsum[0] / (N * H * W) - mean * mean;

    __syncthreads();

    // Step 2: Normalize and scale
    for (int idx = tid; idx < N * H * W; idx += stride) {
        int n = idx / (H * W);
        int hw = idx % (H * W);
        int h = hw / W;
        int w = hw % W;

        int offset = ((n * C + c) * H + h) * W + w;
        float x = input[offset];
        float x_hat = (x - mean) / sqrtf(var + eps);
        output[offset] = gamma[c] * x_hat + beta[c];
    }
}

template <int TILE>
__global__ void batch_norm_4d_backward(const float* __restrict__ doutput,
                                       const float* __restrict__ input,
                                       float* __restrict__ dinput,
                                       const float* __restrict__ gamma,
                                       float* __restrict__ dgamma,
                                       float* __restrict__ dbeta,
                                       int N, int C, int H, int W,
                                       float eps) {
    int c = blockIdx.x;
    if (c >= C) return;

    __shared__ float s_data[TILE * 2];
    
    __shared__ float s_scalars[4]; 

    int tid = threadIdx.x;
    int stride = blockDim.x;
    float count = static_cast<float>(N * H * W);

    // === PASS 1: Calculate mean and variance ===
    // Same as the forward pass kernel.
    float p1_sum = 0.0f;
    float p1_sqsum = 0.0f;

    for (int i = tid; i < N * H * W; i += stride) {
        int n = i / (H * W);
        int hw = i % (H * W);
        int h = hw / W;
        int w = hw % W;
        int index = ((n * C + c) * H + h) * W + w;
        float val = input[index];
        p1_sum += val;
        p1_sqsum += val * val;
    }

    s_data[tid] = p1_sum;
    s_data[tid + TILE] = p1_sqsum;
    __syncthreads();

    for (int s = TILE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
            s_data[tid + TILE] += s_data[tid + TILE + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mean = s_data[0] / count;
        float var = (s_data[TILE] / count) - mean * mean;
        s_scalars[0] = mean;
        s_scalars[1] = var;
    }
    __syncthreads();

    float mean = s_scalars[0];
    float var = s_scalars[1];
    float std_inv = rsqrtf(var + eps);

    // === PASS 2: Calculate dgamma and dbeta ===
    float p2_dbeta = 0.0f;
    float p2_dgamma = 0.0f;

    for (int i = tid; i < N * H * W; i += stride) {
        int n = i / (H * W);
        int hw = i % (H * W);
        int h = hw / W;
        int w = hw % W;
        int index = ((n * C + c) * H + h) * W + w;

        float x_hat = (input[index] - mean) * std_inv;
        p2_dbeta += doutput[index];
        p2_dgamma += doutput[index] * x_hat;
    }

    s_data[tid] = p2_dgamma;
    s_data[tid + TILE] = p2_dbeta;
    __syncthreads();

    for (int s = TILE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
            s_data[tid + TILE] += s_data[tid + TILE + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        dgamma[c] = s_data[0];
        dbeta[c] = s_data[TILE];
        s_scalars[2] = s_data[0];
        s_scalars[3] = s_data[TILE];
    }
    __syncthreads();

    float final_dgamma = s_scalars[2];
    float final_dbeta = s_scalars[3];

    // === PASS 3: Calculate dinput ===
    float common_term = gamma[c] * std_inv;

    for (int i = tid; i < N * H * W; i += stride) {
        int n = i / (H * W);
        int hw = i % (H * W);
        int h = hw / W;
        int w = hw % W;
        int index = ((n * C + c) * H + h) * W + w;

        float x_hat = (input[index] - mean) * std_inv;
        
        float paren_term = count * doutput[index] - final_dbeta - x_hat * final_dgamma;
        
        dinput[index] = (common_term / count) * paren_term;
    }
}

void batch_norm_4d_cuda_launcher(const float* input, float* output,
                                 const float* gamma, const float* beta,
                                 int N, int C, int H, int W, float eps) {
    // Each block processes one channel
    dim3 grid(C);
    dim3 block(TILE_DIM);

    batch_norm_4d<TILE_DIM><<<grid, block>>>(input, output, gamma, beta, N, C, H, W, eps);
}

void batch_norm_4d_backward_cuda_launcher(const float* doutput, const float* input,
                                          float* dinput, const float* gamma,
                                          float* dgamma, float* dbeta,
                                          int N, int C, int H, int W, float eps) {
    // Each block processes one channel
    dim3 grid(C);
    dim3 block(TILE_DIM);

    batch_norm_4d_backward<TILE_DIM><<<grid, block>>>(doutput, input, dinput, gamma, dgamma, dbeta, N, C, H, W, eps);
}


void batch_norm_4d_cpu(const float* input,
                       float* output,
                       const float* gamma,
                       const float* beta,
                       int N, int C, int H, int W,
                       float eps) {
    for (int c = 0; c < C; ++c) {
        float sum = 0.0f;
        float sqsum = 0.0f;
        for (int n = 0; n < N; ++n) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    int index = ((n * C + c) * H + h) * W + w;
                    float val = input[index];
                    sum += val;
                    sqsum += val * val;
                }
            }
        }

        float count = static_cast<float>(N * H * W);
        float mean = sum / count;
        float var = sqsum / count - mean * mean;

        for (int n = 0; n < N; ++n) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    int index = ((n * C + c) * H + h) * W + w;
                    float x = input[index];
                    float x_hat = (x - mean) / sqrtf(var + eps);
                    output[index] = gamma[c] * x_hat + beta[c];
                }
            }
        }
    }
}

