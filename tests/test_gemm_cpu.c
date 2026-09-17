#include <stdio.h>
#include <stdlib.h>

#include "gemm.h"
#include "matrix.h"

typedef struct {
    size_t m;
    size_t n;
    size_t k;
} GemmShape;

static void gemm_long_double_reference(const float *a,
                                       const float *b,
                                       float *c,
                                       size_t m,
                                       size_t n,
                                       size_t k)
{
    for (size_t row = 0; row < m; ++row) {
        for (size_t col = 0; col < n; ++col) {
            long double sum = 0.0L;

            for (size_t inner = 0; inner < k; ++inner) {
                sum += (long double)a[row * k + inner]
                     * (long double)b[inner * n + col];
            }

            c[row * n + col] = (float)sum;
        }
    }
}

static int test_hand_computed_example(void)
{
    const float a[2 * 3] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    const float b[3 * 2] = {
         7.0f,  8.0f,
         9.0f, 10.0f,
        11.0f, 12.0f
    };

    const float expected[2 * 2] = {
         58.0f,  64.0f,
        139.0f, 154.0f
    };

    float result[2 * 2] = {0.0f};

    gemm_cpu(a, b, result, 2, 2, 3);

    for (size_t index = 0; index < 2 * 2; ++index) {
        if (result[index] != expected[index]) {
            fprintf(stderr,
                    "element %zu: expected %.1f, obtained %.1f\n",
                    index, expected[index], result[index]);
            return 0;
        }
    }

    return 1;
}

static int test_random_shape(GemmShape shape, uint32_t seed)
{
    float *a = matrix_alloc(shape.m, shape.k);
    float *b = matrix_alloc(shape.k, shape.n);
    float *actual = matrix_alloc(shape.m, shape.n);
    float *reference = matrix_alloc(shape.m, shape.n);

    if (a == NULL || b == NULL || actual == NULL || reference == NULL) {
        fputs("matrix allocation failed during GEMM test\n", stderr);
        matrix_free(a);
        matrix_free(b);
        matrix_free(actual);
        matrix_free(reference);
        return 0;
    }

    matrix_fill_random(a, shape.m, shape.k, seed);
    matrix_fill_random(b, shape.k, shape.n, seed + UINT32_C(1));

    gemm_cpu(a, b, actual, shape.m, shape.n, shape.k);
    gemm_long_double_reference(a, b, reference,
                               shape.m, shape.n, shape.k);

    const MatrixComparison comparison =
        matrix_compare(actual, reference, shape.m, shape.n,
                       1.0e-6, 1.0e-6);

    int passed = comparison.valid && comparison.mismatch_count == 0;

    if (!passed) {
        const size_t row = comparison.worst_index / shape.n;
        const size_t col = comparison.worst_index % shape.n;

        fprintf(stderr,
                "GEMM %zux%zux%zu failed at (%zu, %zu): "
                "absolute error %.9g, allowed %.9g\n",
                shape.m, shape.n, shape.k, row, col,
                comparison.worst_absolute_error,
                comparison.worst_allowed_error);
    }

    matrix_free(a);
    matrix_free(b);
    matrix_free(actual);
    matrix_free(reference);
    return passed;
}

int main(void)
{
    if (!test_hand_computed_example()) {
        return EXIT_FAILURE;
    }

    const GemmShape shapes[] = {
        {1, 1, 1},
        {2, 5, 3},
        {3, 2, 7},
        {15, 17, 19},
        {31, 33, 32}
    };

    const size_t shape_count = sizeof(shapes) / sizeof(shapes[0]);
    for (size_t index = 0; index < shape_count; ++index) {
        if (!test_random_shape(shapes[index],
                               UINT32_C(1000) + (uint32_t)index)) {
            return EXIT_FAILURE;
        }
    }

    puts("CPU GEMM test passed.");
    return EXIT_SUCCESS;
}

