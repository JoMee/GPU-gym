#include "gemm.h"

void gemm_cpu(const float *a,
              const float *b,
              float *c,
              size_t m,
              size_t n,
              size_t k)
{
    for (size_t row = 0; row < m; ++row) {
        for (size_t col = 0; col < n; ++col) {
            double sum = 0.0;

            for (size_t inner = 0; inner < k; ++inner) {
                sum += (double)a[row * k + inner]
                     * (double)b[inner * n + col];
            }

            c[row * n + col] = (float)sum;
        }
    }
}


