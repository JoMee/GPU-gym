#include "gemm_cuda.h"

#include <cuda_runtime.h>

__global__ static void gemm_naive_kernel(const float *a,
                                         const float *b,
                                         float *c,
                                         size_t m,
                                         size_t n,
                                         size_t k)
{
    const size_t row = (size_t)blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || col >= n) {
        return;
    }

    float sum = 0.0f;

    for (size_t inner = 0; inner < k; ++inner) {
        sum += a[row * k + inner] * b[inner * n + col];
    }

    c[row * n + col] = sum;
}

extern "C"
cudaError_t gemm_cuda_naive_launch(const float *device_a,
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

    const dim3 block(16, 16);
    const dim3 grid((unsigned int)((n + block.x - 1) / block.x),
                    (unsigned int)((m + block.y - 1) / block.y));

    gemm_naive_kernel<<<grid, block>>>(device_a, device_b, device_c,
                                      m, n, k);
    return cudaGetLastError();
}


