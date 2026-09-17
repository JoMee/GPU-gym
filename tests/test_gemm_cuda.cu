#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#include "cuda_check.h"
#include "gemm.h"
#include "gemm_cuda.h"
#include "matrix.h"

typedef struct {
    size_t m;
    size_t n;
    size_t k;
} GemmShape;

typedef cudaError_t (*GemmLaunch)(const float *, const float *, float *,
                                  size_t, size_t, size_t);

typedef struct {
    const char *name;
    GemmLaunch launch;
} GemmImplementation;

static int test_shape(GemmImplementation implementation,
                      GemmShape shape,
                      uint32_t seed)
{
    const size_t a_bytes = shape.m * shape.k * sizeof(float);
    const size_t b_bytes = shape.k * shape.n * sizeof(float);
    const size_t c_bytes = shape.m * shape.n * sizeof(float);

    float *a = matrix_alloc(shape.m, shape.k);
    float *b = matrix_alloc(shape.k, shape.n);
    float *reference = matrix_alloc(shape.m, shape.n);
    float *actual = matrix_alloc(shape.m, shape.n);

    if (a == NULL || b == NULL || reference == NULL || actual == NULL) {
        fputs("host allocation failed during CUDA GEMM test\n", stderr);
        matrix_free(a);
        matrix_free(b);
        matrix_free(reference);
        matrix_free(actual);
        return 0;
    }

    matrix_fill_random(a, shape.m, shape.k, seed);
    matrix_fill_random(b, shape.k, shape.n, seed + UINT32_C(1));
    gemm_cpu(a, b, reference, shape.m, shape.n, shape.k);

    float *device_a = NULL;
    float *device_b = NULL;
    float *device_c = NULL;

    CUDA_CHECK(cudaMalloc((void **)&device_a, a_bytes));
    CUDA_CHECK(cudaMalloc((void **)&device_b, b_bytes));
    CUDA_CHECK(cudaMalloc((void **)&device_c, c_bytes));

    CUDA_CHECK(cudaMemcpy(device_a, a, a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_b, b, b_bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(implementation.launch(device_a, device_b, device_c,
                                     shape.m, shape.n, shape.k));

    /* This blocking copy also waits for the kernel and reports execution errors. */
    CUDA_CHECK(cudaMemcpy(actual, device_c, c_bytes, cudaMemcpyDeviceToHost));

    const MatrixComparison comparison =
        matrix_compare(actual, reference, shape.m, shape.n,
                       1.0e-5, 1.0e-5);

    const int passed = comparison.valid && comparison.mismatch_count == 0;

    if (!passed) {
        const size_t row = comparison.worst_index / shape.n;
        const size_t col = comparison.worst_index % shape.n;

        fprintf(stderr,
                "%s CUDA GEMM %zux%zux%zu failed at (%zu, %zu): "
                "GPU %.9g, CPU %.9g, absolute error %.9g, allowed %.9g\n",
                implementation.name, shape.m, shape.n, shape.k, row, col,
                actual[comparison.worst_index],
                reference[comparison.worst_index],
                comparison.worst_absolute_error,
                comparison.worst_allowed_error);
    }

    CUDA_CHECK(cudaFree(device_a));
    CUDA_CHECK(cudaFree(device_b));
    CUDA_CHECK(cudaFree(device_c));
    matrix_free(a);
    matrix_free(b);
    matrix_free(reference);
    matrix_free(actual);
    return passed;
}

int main(void)
{
    const GemmImplementation implementations[] = {
        {"naive", gemm_cuda_naive_launch},
        {"tiled", gemm_cuda_tiled_launch},
        {"register_tiled_2x2", gemm_cuda_register_tiled_launch}
    };

    const GemmShape shapes[] = {
        {1, 1, 1},
        {2, 5, 3},
        {3, 2, 7},
        {15, 17, 19},
        {31, 33, 32},
        {33, 31, 35}
    };

    const size_t implementation_count =
        sizeof(implementations) / sizeof(implementations[0]);
    const size_t shape_count = sizeof(shapes) / sizeof(shapes[0]);

    for (size_t implementation = 0;
         implementation < implementation_count;
         ++implementation) {
        for (size_t index = 0; index < shape_count; ++index) {
            if (!test_shape(implementations[implementation], shapes[index],
                            UINT32_C(2000) + (uint32_t)index)) {
                return EXIT_FAILURE;
            }
        }
    }

    puts("Naive, tiled, and 2x2 register-tiled CUDA GEMM tests passed.");
    return EXIT_SUCCESS;
}

