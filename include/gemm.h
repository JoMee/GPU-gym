#ifndef GEMM_H
#define GEMM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Compute C = A B using row-major storage.
 *
 * A has shape m x k.
 * B has shape k x n.
 * C has shape m x n and is overwritten.
 */
void gemm_cpu(const float *a,
              const float *b,
              float *c,
              size_t m,
              size_t n,
              size_t k);

#ifdef __cplusplus
}
#endif

#endif


