#include <cuda_runtime.h>
#include <iostream>


#define TILE_WIDTH 16
#define KERNEL_RADIUS 1
#define KERNEL_SIZE (2 * KERNEL_RADIUS + 1)
#define SHARED_WIDTH (TILE_WIDTH + 2 * KERNEL_RADIUS)


__constant__ float kernel[KERNEL_SIZE * KERNEL_SIZE];



__global__ void conv2d_4d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int B, int C, int H, int W
) {

    __shared__ float tile[SHARED_WIDTH][SHARED_WIDTH];

    int tx_load = threadIdx.x;
    int ty_load = threadIdx.y;
    int load_x = blockIdx.x * TILE_WIDTH + tx_load - KERNEL_RADIUS;
    int load_y = blockIdx.y * TILE_WIDTH + ty_load - KERNEL_RADIUS;
    int c = blockIdx.z % C;
    int b = blockIdx.z / C;


    float val = 0.0f;
    if (load_x >= 0 && load_x < W && load_y >= 0 && load_y < H) {
        val = input[((b * C + c) * H + load_y) * W + load_x];
    }
    tile[ty_load][tx_load] = val;

    __syncthreads();


    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if (tx < TILE_WIDTH && ty < TILE_WIDTH) {
        int out_x = blockIdx.x * TILE_WIDTH + tx;
        int out_y = blockIdx.y * TILE_WIDTH + ty;

        if (out_x < W && out_y < H) {
            float sum = 0.0f;
            // Perform the convolution using the cached data in shared memory.
            for (int ky = 0; ky < KERNEL_SIZE; ++ky) {
                for (int kx = 0; kx < KERNEL_SIZE; ++kx) {
                    sum += tile[ty + ky][tx + kx] * kernel[ky * KERNEL_SIZE + kx];
                }
            }
            // Write the result to global memory.
            int global_idx = ((b * C + c) * H + out_y) * W + out_x;
            output[global_idx] = sum;
        }
    }
}



__global__ void conv2d_4d_input_grad_kernel(
    const float* __restrict__ dOutput,
    float* __restrict__ dInput,
    int B, int C, int H, int W
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int out_x = blockIdx.x * TILE_WIDTH + tx;
    int out_y = blockIdx.y * TILE_WIDTH + ty;
    int c = blockIdx.z % C;
    int b = blockIdx.z / C;

    if (out_x < W && out_y < H) {
        float sum = 0.0f;
        // Convolve dOutput with the 180-degree rotated kernel.
        for (int ky = 0; ky < KERNEL_SIZE; ++ky) {
            for (int kx = 0; kx < KERNEL_SIZE; ++kx) {
                int in_y = out_y + ky - KERNEL_RADIUS;
                int in_x = out_x + kx - KERNEL_RADIUS;
                if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                    int offset = ((b * C + c) * H + in_y) * W + in_x;
                    // Access the kernel with rotated indices.
                    float k_val = kernel[(KERNEL_SIZE - 1 - ky) * KERNEL_SIZE + (KERNEL_SIZE - 1 - kx)];
                    sum += dOutput[offset] * k_val;
                }
            }
        }
        int idx = ((b * C + c) * H + out_y) * W + out_x;
        dInput[idx] = sum;
    }
}



__global__ void conv2d_4d_kernel_grad_kernel(
    const float* __restrict__ input,
    const float* __restrict__ dOutput,
    float* __restrict__ dKernel,
    int B, int C, int H, int W
) {
    // Each thread computes one element of the kernel gradient.
    int ky = threadIdx.y;
    int kx = threadIdx.x;

    float grad = 0.0f;
    // Iterate over the entire batch, channels, and spatial dimensions.
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    int in_y = y - KERNEL_RADIUS + ky;
                    int in_x = x - KERNEL_RADIUS + kx;

                    if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                        int input_idx  = ((b * C + c) * H + in_y) * W + in_x;
                        int output_idx = ((b * C + c) * H + y) * W + x;
                        grad += input[input_idx] * dOutput[output_idx];
                    }
                }
            }
        }
    }

    int kernel_idx = ky * KERNEL_SIZE + kx;
    atomicAdd(&dKernel[kernel_idx], grad);
}


// --- CUDA Launcher Functions ---

extern "C" void conv2d_4d_forward_cuda_launcher(const float* input, float* output, const float* host_kernel,
                                              int B, int C, int H, int W) {
    // Copy the kernel from host memory to device constant memory
    cudaMemcpyToSymbol(kernel, host_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));

    dim3 grid((W + TILE_WIDTH - 1) / TILE_WIDTH, (H + TILE_WIDTH - 1) / TILE_WIDTH, B * C);
    dim3 block(SHARED_WIDTH, SHARED_WIDTH);

    conv2d_4d_forward_kernel<<<grid, block>>>(input, output, B, C, H, W);
}

extern "C" void conv2d_4d_input_grad_cuda_launcher(const float* dOutput, float* dInput, const float* host_kernel,
                                                 int B, int C, int H, int W) {
    // Copy the kernel from host memory to device constant memory
    cudaMemcpyToSymbol(kernel, host_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));

    dim3 grid((W + TILE_WIDTH - 1) / TILE_WIDTH, (H + TILE_WIDTH - 1) / TILE_WIDTH, B * C);
    dim3 block(TILE_WIDTH, TILE_WIDTH);

    conv2d_4d_input_grad_kernel<<<grid, block>>>(dOutput, dInput, B, C, H, W);
}

extern "C" void conv2d_4d_kernel_grad_cuda_launcher(const float* input, const float* dOutput, float* dKernel,
                                                  int B, int C, int H, int W) {
    cudaMemset(dKernel, 0, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));

    dim3 grid(1, 1, 1);
    dim3 block(KERNEL_SIZE, KERNEL_SIZE);

    conv2d_4d_kernel_grad_kernel<<<grid, block>>>(input, dOutput, dKernel, B, C, H, W);
}
