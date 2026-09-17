#include "gemm_cuda.h"

#include <cuda_runtime.h>

enum { TILE_WIDTH = 16 };

__global__ static void gemm_tiled_kernel(const float *a,
                                         const float *b,
                                         float *c,
                                         size_t m,
                                         size_t n,
                                         size_t k)
{
    __shared__ float tile_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_b[TILE_WIDTH][TILE_WIDTH];

    const size_t row = (size_t)blockIdx.y * TILE_WIDTH + threadIdx.y;
    const size_t col = (size_t)blockIdx.x * TILE_WIDTH + threadIdx.x;
    const size_t tile_count = (k - 1) / TILE_WIDTH + 1;

    float sum = 0.0f;

    for (size_t tile = 0; tile < tile_count; ++tile) {
        const size_t a_col = tile * TILE_WIDTH + threadIdx.x;
        const size_t b_row = tile * TILE_WIDTH + threadIdx.y;

        tile_a[threadIdx.y][threadIdx.x] =
            row < m && a_col < k ? a[row * k + a_col] : 0.0f;
        tile_b[threadIdx.y][threadIdx.x] =
            b_row < k && col < n ? b[b_row * n + col] : 0.0f;

        __syncthreads();

#pragma unroll
        for (size_t inner = 0; inner < TILE_WIDTH; ++inner) {
            sum += tile_a[threadIdx.y][inner]
                 * tile_b[inner][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = sum;
    }
}

extern "C"
cudaError_t gemm_cuda_tiled_launch(const float *device_a,
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

    const dim3 block(TILE_WIDTH, TILE_WIDTH);
    const dim3 grid((unsigned int)((n + TILE_WIDTH - 1) / TILE_WIDTH),
                    (unsigned int)((m + TILE_WIDTH - 1) / TILE_WIDTH));

    gemm_tiled_kernel<<<grid, block>>>(device_a, device_b, device_c,
                                      m, n, k);
    return cudaGetLastError();
}
