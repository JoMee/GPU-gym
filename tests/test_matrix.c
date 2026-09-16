#include <stdint.h>
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

int main(void)
{
    if (!test_allocation_and_fill()
        || !test_invalid_sizes()
        || !test_deterministic_random_fill()) {
        return EXIT_FAILURE;
    }

    puts("Matrix utility tests passed.");
    return EXIT_SUCCESS;
}


