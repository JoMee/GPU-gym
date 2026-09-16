#include "matrix.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int matrix_element_count(size_t rows,
                                size_t cols,
                                size_t *element_count)
{
    if (rows == 0 || cols == 0 || element_count == NULL) {
        return 0;
    }

    if (rows > SIZE_MAX / cols) {
        return 0;
    }

    *element_count = rows * cols;
    return 1;
}

static uint32_t xorshift32(uint32_t *state)
{
    uint32_t value = *state;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    *state = value;
    return value;
}

float *matrix_alloc(size_t rows, size_t cols)
{
    size_t element_count = 0;

    if (!matrix_element_count(rows, cols, &element_count)) {
        return NULL;
    }

    if (element_count > SIZE_MAX / sizeof(float)) {
        return NULL;
    }

    return malloc(element_count * sizeof(float));
}

void matrix_free(float *matrix)
{
    free(matrix);
}

void matrix_fill(float *matrix,
                 size_t rows,
                 size_t cols,
                 float value)
{
    size_t element_count = 0;

    if (matrix == NULL
        || !matrix_element_count(rows, cols, &element_count)) {
        return;
    }

    for (size_t index = 0; index < element_count; ++index) {
        matrix[index] = value;
    }
}

void matrix_fill_random(float *matrix,
                        size_t rows,
                        size_t cols,
                        uint32_t seed)
{
    size_t element_count = 0;

    if (matrix == NULL
        || !matrix_element_count(rows, cols, &element_count)) {
        return;
    }

    uint32_t state = seed == 0 ? UINT32_C(0x9e3779b9) : seed;

    for (size_t index = 0; index < element_count; ++index) {
        const uint32_t random_bits = xorshift32(&state) >> 8;
        const float unit = (float)random_bits * (1.0f / 16777216.0f);
        matrix[index] = 2.0f * unit - 1.0f;
    }
}

void matrix_print(const char *name,
                  const float *matrix,
                  size_t rows,
                  size_t cols)
{
    if (matrix == NULL) {
        return;
    }

    if (name != NULL) {
        printf("%s (%zu x %zu):\n", name, rows, cols);
    }

    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            printf("%10.4f", matrix[row * cols + col]);
        }
        putchar('\n');
    }
}


