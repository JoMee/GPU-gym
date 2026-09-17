#include "gemm_cuda.h"

#include <cuda_runtime.h>

enum {
    THREAD_TILE_WIDTH = 16,
    OUTPUTS_PER_THREAD = 4,
    OUTPUT_TILE_WIDTH = THREAD_TILE_WIDTH * OUTPUTS_PER_THREAD
};

__global__ static void gemm_register_tiled_kernel(const float *a,
                                                  const float *b,
                                                  float *c,
                                                  size_t m,
                                                  size_t n,
                                                  size_t k)
{
    __shared__ float tile_a[OUTPUT_TILE_WIDTH][THREAD_TILE_WIDTH];
    __shared__ float tile_b[THREAD_TILE_WIDTH][OUTPUT_TILE_WIDTH];

    const size_t output_row =
        (size_t)blockIdx.y * OUTPUT_TILE_WIDTH + threadIdx.y;
    const size_t output_col =
        (size_t)blockIdx.x * OUTPUT_TILE_WIDTH + threadIdx.x;
    const size_t second_row = output_row + THREAD_TILE_WIDTH;
    const size_t second_col = output_col + THREAD_TILE_WIDTH;
    const size_t tile_count = (k - 1) / THREAD_TILE_WIDTH + 1;

    float sum00 = 0.0f;
    float sum01 = 0.0f;
    float sum10 = 0.0f;
    float sum11 = 0.0f;

    for (size_t tile = 0; tile < tile_count; ++tile) {
        const size_t a_col = tile * THREAD_TILE_WIDTH + threadIdx.x;
        const size_t b_row = tile * THREAD_TILE_WIDTH + threadIdx.y;

        tile_a[threadIdx.y][threadIdx.x] =
            output_row < m && a_col < k
                ? a[output_row * k + a_col]
                : 0.0f;
        tile_a[threadIdx.y + THREAD_TILE_WIDTH][threadIdx.x] =
            second_row < m && a_col < k
                ? a[second_row * k + a_col]
                : 0.0f;

        tile_b[threadIdx.y][threadIdx.x] =
            b_row < k && output_col < n
                ? b[b_row * n + output_col]
                : 0.0f;
        tile_b[threadIdx.y][threadIdx.x + THREAD_TILE_WIDTH] =
            b_row < k && second_col < n
                ? b[b_row * n + second_col]
                : 0.0f;

        __syncthreads();

#pragma unroll
        for (size_t inner = 0; inner < THREAD_TILE_WIDTH; ++inner) {
            const float a0 = tile_a[threadIdx.y][inner];
            const float a1 =
                tile_a[threadIdx.y + THREAD_TILE_WIDTH][inner];
            const float b0 = tile_b[inner][threadIdx.x];
            const float b1 =
                tile_b[inner][threadIdx.x + THREAD_TILE_WIDTH];

            sum00 += a0 * b0;
            sum01 += a0 * b1;
            sum10 += a1 * b0;
            sum11 += a1 * b1;
        }

        __syncthreads();
    }

    if (output_row < m && output_col < n) {
        c[output_row * n + output_col] = sum00;
    }
    if (output_row < m && second_col < n) {
        c[output_row * n + second_col] = sum01;
    }
    if (second_row < m && output_col < n) {
        c[second_row * n + output_col] = sum10;
    }
    if (second_row < m && second_col < n) {
        c[second_row * n + second_col] = sum11;
    }
}

extern "C"
cudaError_t gemm_cuda_register_tiled_launch(const float *device_a,
                                            const float *device_b,
                                            float *device_c,
                                            size_t m,
                                            size_t n,
                                            size_t k)
{
    if (device_a == NULL
        || device_b == NULL
        || device_c == NULL
        || m == 0
        || n == 0
        || k == 0) {
        return cudaErrorInvalidValue;
    }

    const dim3 block(THREAD_TILE_WIDTH, THREAD_TILE_WIDTH);
    const dim3 grid(
        (unsigned int)((n - 1) / OUTPUT_TILE_WIDTH + 1),
        (unsigned int)((m - 1) / OUTPUT_TILE_WIDTH + 1));

    gemm_register_tiled_kernel<<<grid, block>>>(device_a, device_b, device_c,
                                               m, n, k);
    return cudaGetLastError();
}
