#ifndef CUDA_CHECK_H
#define CUDA_CHECK_H

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t cuda_status__ = (call);                                \
        if (cuda_status__ != cudaSuccess) {                                \
            fprintf(stderr, "%s:%d: CUDA error: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(cuda_status__));\
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

#endif


