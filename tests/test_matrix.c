#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"

static int test_allocation_and_fill(void)
{
    float *matrix = matrix_alloc(2, 3);
    if (matrix == NULL) {
        fputs("matrix_alloc unexpectedly failed\n", stderr);
        return 0;
    }

    matrix_fill(matrix, 2, 3, 3.25f);

    for (size_t index = 0; index < 2 * 3; ++index) {
        if (matrix[index] != 3.25f) {
            fprintf(stderr, "matrix_fill failed at element %zu\n", index);
            matrix_free(matrix);
            return 0;
        }
    }

    matrix_free(matrix);
    return 1;
}

static int test_invalid_sizes(void)
{
    if (matrix_alloc(0, 4) != NULL) {
        fputs("zero-sized allocation should fail\n", stderr);
        return 0;
    }

    if (matrix_alloc(SIZE_MAX, 2) != NULL) {
        fputs("overflowing allocation should fail\n", stderr);
        return 0;
    }

    return 1;
}

static int test_deterministic_random_fill(void)
{
    float first[16];
    float second[16];
    float different[16];

    matrix_fill_random(first, 4, 4, UINT32_C(12345));
    matrix_fill_random(second, 4, 4, UINT32_C(12345));
    matrix_fill_random(different, 4, 4, UINT32_C(54321));

    int found_difference = 0;

    for (size_t index = 0; index < 16; ++index) {
        if (first[index] != second[index]) {
            fprintf(stderr,
                    "same seed differed at element %zu\n",
                    index);
            return 0;
        }

        if (first[index] < -1.0f || first[index] >= 1.0f) {
            fprintf(stderr,
                    "random value out of range at element %zu\n",
                    index);
            return 0;
        }

        if (first[index] != different[index]) {
            found_difference = 1;
        }
    }

    if (!found_difference) {
        fputs("different seeds unexpectedly produced equal matrices\n", stderr);
        return 0;
    }

    return 1;
}

static int test_matrix_comparison(void)
{
    const float reference[3] = {0.0f, 100.0f, -2.0f};
    const float actual[3] = {1.0e-7f, 100.05f, -2.1f};

    const MatrixComparison comparison =
        matrix_compare(actual, reference, 1, 3, 1.0e-6, 1.0e-3);

    if (!comparison.valid) {
        fputs("valid matrix comparison was rejected\n", stderr);
        return 0;
    }

    if (comparison.mismatch_count != 1 || comparison.worst_index != 2) {
        fprintf(stderr,
                "expected one mismatch at index 2, got %zu at index %zu\n",
                comparison.mismatch_count,
                comparison.worst_index);
        return 0;
    }

    const float nonfinite[3] = {0.0f, NAN, -2.0f};
    const MatrixComparison nan_comparison =
        matrix_compare(nonfinite, reference, 1, 3, 0.0, 0.0);

    if (!nan_comparison.valid || nan_comparison.mismatch_count != 1) {
        fputs("NaN comparison should produce one mismatch\n", stderr);
        return 0;
    }

    const MatrixComparison invalid =
        matrix_compare(NULL, reference, 1, 3, 0.0, 0.0);

    if (invalid.valid) {
        fputs("NULL comparison should be invalid\n", stderr);
        return 0;
    }

    return 1;
}

int main(void)
{
    if (!test_allocation_and_fill()
        || !test_invalid_sizes()
        || !test_deterministic_random_fill()
        || !test_matrix_comparison()) {
        return EXIT_FAILURE;
    }

    puts("Matrix utility tests passed.");
    return EXIT_SUCCESS;
}
