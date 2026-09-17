#ifndef GEMM_CUDA_H
#define GEMM_CUDA_H

#include <stddef.h>

#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Launch C = A B on the default stream using row-major device arrays.
 *
 * A has shape m x k.
 * B has shape k x n.
 * C has shape m x n and is overwritten.
 *
 * The launch is asynchronous. This function reports launch errors only;
 * execution errors are reported by a later synchronization or blocking copy.
 */
cudaError_t gemm_cuda_naive_launch(const float *device_a,
                                   const float *device_b,
                                   float *device_c,
                                   size_t m,
                                   size_t n,
                                   size_t k);

/* Same operation, using 16 x 16 tiles staged explicitly in shared memory. */
cudaError_t gemm_cuda_tiled_launch(const float *device_a,
                                   const float *device_b,
                                   float *device_c,
                                   size_t m,
                                   size_t n,
                                   size_t k);

#ifdef __cplusplus
}
#endif

#endif

