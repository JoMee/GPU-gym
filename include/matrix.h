#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int valid;
    size_t mismatch_count;
    size_t worst_index;
    double worst_absolute_error;
    double worst_allowed_error;
    double worst_error_ratio;
} MatrixComparison;

/*
 * Allocate a row-major rows-by-cols matrix.
 *
 * Returns NULL if either dimension is zero, if the requested byte count
 * overflows size_t, or if allocation fails.
 */
float *matrix_alloc(size_t rows, size_t cols);

void matrix_free(float *matrix);

void matrix_fill(float *matrix,
                 size_t rows,
                 size_t cols,
                 float value);

/* Fill a matrix deterministically with values in [-1, 1). */
void matrix_fill_random(float *matrix,
                        size_t rows,
                        size_t cols,
                        uint32_t seed);

void matrix_print(const char *name,
                  const float *matrix,
                  size_t rows,
                  size_t cols);

/*
 * Compare actual against reference elementwise using
 *
 *   |actual - reference| <= absolute_tolerance
 *                          + relative_tolerance * |reference|.
 *
 * The comparison is invalid if a pointer is NULL, a dimension is zero,
 * the element count overflows, or either tolerance is negative.
 */
MatrixComparison matrix_compare(const float *actual,
                                const float *reference,
                                size_t rows,
                                size_t cols,
                                double absolute_tolerance,
                                double relative_tolerance);

#ifdef __cplusplus
}
#endif

#endif
