#include <stdio.h>
#include <stdlib.h>

#include "gemm.h"

int main(void)
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
            return EXIT_FAILURE;
        }
    }

    puts("CPU GEMM test passed.");
    return EXIT_SUCCESS;
}


