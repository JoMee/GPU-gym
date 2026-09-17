#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda_runtime.h>

#include "cuda_check.h"
#include "gemm_cuda.h"
#include "matrix.h"

enum {
    WARMUP_LAUNCHES = 10,
    SAMPLE_COUNT = 15,
    CALIBRATION_ITERATIONS = 3,
    MAX_KERNEL_ITERATIONS = 10000,
    MAX_PIPELINE_ITERATIONS = 1000
};

static const double TARGET_SAMPLE_MS = 100.0;

typedef struct {
    size_t m;
    size_t n;
    size_t k;
} GemmShape;

typedef struct {
    double minimum;
    double median;
    double p90;
} Statistics;

typedef cudaError_t (*GemmLaunch)(const float *, const float *, float *,
                                  size_t, size_t, size_t);

typedef struct {
    const char *name;
    GemmLaunch launch;
} GemmImplementation;

static int compare_double(const void *left, const void *right)
{
    const double a = *(const double *)left;
    const double b = *(const double *)right;
    return (a > b) - (a < b);
}

static Statistics compute_statistics(double *samples, size_t sample_count)
{
    qsort(samples, sample_count, sizeof(samples[0]), compare_double);

    Statistics result;
    result.minimum = samples[0];

    if (sample_count % 2 == 0) {
        const size_t upper = sample_count / 2;
        result.median = 0.5 * (samples[upper - 1] + samples[upper]);
    } else {
        result.median = samples[sample_count / 2];
    }

    const size_t p90_index = (90 * (sample_count - 1) + 50) / 100;
    result.p90 = samples[p90_index];
    return result;
}

static int matrix_bytes(size_t rows, size_t cols, size_t *bytes)
{
    if (rows == 0 || cols == 0 || bytes == NULL) {
        return 0;
    }

    if (rows > SIZE_MAX / cols) {
        return 0;
    }

    const size_t elements = rows * cols;
    if (elements > SIZE_MAX / sizeof(float)) {
        return 0;
    }

    *bytes = elements * sizeof(float);
    return 1;
}

static double monotonic_ms(void)
{
    struct timespec timestamp;
    if (clock_gettime(CLOCK_MONOTONIC, &timestamp) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }

    return 1000.0 * (double)timestamp.tv_sec
         + 1.0e-6 * (double)timestamp.tv_nsec;
}

static double measure_kernel_ms(GemmLaunch launch,
                                const float *device_a,
                                const float *device_b,
                                float *device_c,
                                GemmShape shape,
                                size_t iterations)
{
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        CUDA_CHECK(launch(device_a, device_b, device_c,
                          shape.m, shape.n, shape.k));
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return (double)total_ms / (double)iterations;
}

static double measure_pipeline_ms(GemmLaunch launch,
                                  const float *host_a,
                                  const float *host_b,
                                  float *host_c,
                                  float *device_a,
                                  float *device_b,
                                  float *device_c,
                                  size_t a_bytes,
                                  size_t b_bytes,
                                  size_t c_bytes,
                                  GemmShape shape,
                                  size_t iterations)
{
    const double start_ms = monotonic_ms();

    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        CUDA_CHECK(cudaMemcpy(device_a, host_a, a_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_b, host_b, b_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(launch(device_a, device_b, device_c,
                          shape.m, shape.n, shape.k));
        CUDA_CHECK(cudaMemcpy(host_c, device_c, c_bytes,
                              cudaMemcpyDeviceToHost));
    }

    const double stop_ms = monotonic_ms();
    return (stop_ms - start_ms) / (double)iterations;
}

static size_t choose_iterations(double milliseconds_per_iteration,
                                size_t maximum_iterations)
{
    if (!(milliseconds_per_iteration > 0.0)
        || !isfinite(milliseconds_per_iteration)) {
        return 1;
    }

    double requested = ceil(TARGET_SAMPLE_MS / milliseconds_per_iteration);
    if (requested < 1.0) {
        requested = 1.0;
    }
    if (requested > (double)maximum_iterations) {
        requested = (double)maximum_iterations;
    }

    return (size_t)requested;
}

static int run_benchmark(GemmImplementation implementation,
                         GemmShape shape,
                         uint32_t seed)
{
    size_t a_bytes;
    size_t b_bytes;
    size_t c_bytes;

    if (!matrix_bytes(shape.m, shape.k, &a_bytes)
        || !matrix_bytes(shape.k, shape.n, &b_bytes)
        || !matrix_bytes(shape.m, shape.n, &c_bytes)) {
        fprintf(stderr, "invalid or overflowing GEMM dimensions\n");
        return 0;
    }

    float *host_a = matrix_alloc(shape.m, shape.k);
    float *host_b = matrix_alloc(shape.k, shape.n);
    float *host_c = matrix_alloc(shape.m, shape.n);

    if (host_a == NULL || host_b == NULL || host_c == NULL) {
        fprintf(stderr, "host allocation failed for %zux%zux%zu\n",
                shape.m, shape.n, shape.k);
        matrix_free(host_a);
        matrix_free(host_b);
        matrix_free(host_c);
        return 0;
    }

    matrix_fill_random(host_a, shape.m, shape.k, seed);
    matrix_fill_random(host_b, shape.k, shape.n, seed + UINT32_C(1));

    float *device_a = NULL;
    float *device_b = NULL;
    float *device_c = NULL;

    CUDA_CHECK(cudaMalloc((void **)&device_a, a_bytes));
    CUDA_CHECK(cudaMalloc((void **)&device_b, b_bytes));
    CUDA_CHECK(cudaMalloc((void **)&device_c, c_bytes));
    CUDA_CHECK(cudaMemcpy(device_a, host_a, a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_b, host_b, b_bytes, cudaMemcpyHostToDevice));

    for (int launch = 0; launch < WARMUP_LAUNCHES; ++launch) {
        CUDA_CHECK(implementation.launch(device_a, device_b, device_c,
                                         shape.m, shape.n, shape.k));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    const double calibration_kernel_ms =
        measure_kernel_ms(implementation.launch,
                          device_a, device_b, device_c, shape,
                          CALIBRATION_ITERATIONS);
    const size_t kernel_iterations =
        choose_iterations(calibration_kernel_ms, MAX_KERNEL_ITERATIONS);

    /* One full calibrated batch lets clocks and caches settle before sampling. */
    (void)measure_kernel_ms(implementation.launch,
                            device_a, device_b, device_c, shape,
                            kernel_iterations);

    double kernel_samples[SAMPLE_COUNT];
    for (size_t sample = 0; sample < SAMPLE_COUNT; ++sample) {
        kernel_samples[sample] =
            measure_kernel_ms(implementation.launch,
                              device_a, device_b, device_c, shape,
                              kernel_iterations);
    }

    const double calibration_pipeline_ms =
        measure_pipeline_ms(implementation.launch,
                            host_a, host_b, host_c,
                            device_a, device_b, device_c,
                            a_bytes, b_bytes, c_bytes, shape, 1);
    const size_t pipeline_iterations =
        choose_iterations(calibration_pipeline_ms, MAX_PIPELINE_ITERATIONS);

    double pipeline_samples[SAMPLE_COUNT];
    for (size_t sample = 0; sample < SAMPLE_COUNT; ++sample) {
        pipeline_samples[sample] =
            measure_pipeline_ms(implementation.launch,
                                host_a, host_b, host_c,
                                device_a, device_b, device_c,
                                a_bytes, b_bytes, c_bytes, shape,
                                pipeline_iterations);
    }

    const Statistics kernel =
        compute_statistics(kernel_samples, SAMPLE_COUNT);
    const Statistics pipeline =
        compute_statistics(pipeline_samples, SAMPLE_COUNT);

    const double operations = 2.0 * (double)shape.m
                            * (double)shape.n
                            * (double)shape.k;
    const double kernel_gflops = operations / (kernel.median * 1.0e6);
    const double pipeline_gflops = operations / (pipeline.median * 1.0e6);

    printf("%s,%zu,%zu,%zu,%zu,%zu,"
           "%.6f,%.6f,%.6f,%.3f,"
           "%.6f,%.6f,%.3f\n",
           implementation.name, shape.m, shape.n, shape.k,
           kernel_iterations, pipeline_iterations,
           kernel.minimum, kernel.median, kernel.p90, kernel_gflops,
           pipeline.median, pipeline.p90, pipeline_gflops);

    CUDA_CHECK(cudaFree(device_a));
    CUDA_CHECK(cudaFree(device_b));
    CUDA_CHECK(cudaFree(device_c));
    matrix_free(host_a);
    matrix_free(host_b);
    matrix_free(host_c);
    return 1;
}

static int parse_dimension(const char *text, size_t *value)
{
    errno = 0;
    char *end = NULL;
    const unsigned long long parsed = strtoull(text, &end, 10);

    if (errno != 0
        || end == text
        || *end != '\0'
        || parsed == 0
        || parsed > (unsigned long long)SIZE_MAX) {
        return 0;
    }

    *value = (size_t)parsed;
    return 1;
}

static void print_environment(void)
{
    int device = 0;
    int driver_version = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));

    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    printf("# gpu=%s\n", properties.name);
    printf("# compute_capability=%d.%d\n",
           properties.major, properties.minor);
    printf("# sm_count=%d\n", properties.multiProcessorCount);
    printf("# cuda_runtime=%d.%d\n",
           CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
    printf("# cuda_driver=%d.%d\n",
           driver_version / 1000, (driver_version % 1000) / 10);
    printf("# warmup_launches=%d\n", WARMUP_LAUNCHES);
    printf("# samples=%d\n", SAMPLE_COUNT);
    printf("# target_sample_ms=%.0f\n", TARGET_SAMPLE_MS);
    puts("implementation,m,n,k,kernel_iterations,pipeline_iterations,"
         "kernel_min_ms,kernel_median_ms,kernel_p90_ms,kernel_gflops,"
         "pipeline_median_ms,pipeline_p90_ms,pipeline_effective_gflops");
}

int main(int argc, char **argv)
{
    const GemmImplementation implementations[] = {
        {"naive", gemm_cuda_naive_launch},
        {"tiled", gemm_cuda_tiled_launch}
    };
    const size_t implementation_count =
        sizeof(implementations) / sizeof(implementations[0]);

    print_environment();

    if (argc == 4) {
        GemmShape shape;
        if (!parse_dimension(argv[1], &shape.m)
            || !parse_dimension(argv[2], &shape.n)
            || !parse_dimension(argv[3], &shape.k)) {
            fprintf(stderr, "usage: %s [M N K]\n", argv[0]);
            return EXIT_FAILURE;
        }

        for (size_t implementation = 0;
             implementation < implementation_count;
             ++implementation) {
            if (!run_benchmark(implementations[implementation], shape,
                               UINT32_C(3000))) {
                return EXIT_FAILURE;
            }
        }
        return EXIT_SUCCESS;
    }

    if (argc != 1) {
        fprintf(stderr, "usage: %s [M N K]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const GemmShape shapes[] = {
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {512, 2048, 256},
        {2048, 512, 256}
    };

    const size_t shape_count = sizeof(shapes) / sizeof(shapes[0]);
    for (size_t index = 0; index < shape_count; ++index) {
        for (size_t implementation = 0;
             implementation < implementation_count;
             ++implementation) {
            if (!run_benchmark(implementations[implementation], shapes[index],
                               UINT32_C(3000) + (uint32_t)index)) {
                return EXIT_FAILURE;
            }
        }
    }

    return EXIT_SUCCESS;
}

