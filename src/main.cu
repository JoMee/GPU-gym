#include <stdio.h>

#include <cuda_runtime.h>

#include "cuda_check.h"

int main(void)
{
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        fprintf(stderr, "No CUDA-capable GPU was found.\n");
        return 1;
    }

    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));

    printf("gemm-lab CUDA smoke test\n");
    printf("device: %s\n", properties.name);
    printf("compute capability: %d.%d\n", properties.major, properties.minor);
    printf("global memory: %.2f GiB\n",
           (double)properties.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}


