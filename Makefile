CC   ?= cc
NVCC ?= nvcc

CUDA_ARCH  ?= native
BUILD_TYPE ?= release

BUILD_DIR := build/$(BUILD_TYPE)
TARGET    := $(BUILD_DIR)/gemm_lab

CPPFLAGS  := -Iinclude
CFLAGS    := -std=c17 -Wall -Wextra -Wpedantic
NVCCFLAGS := -arch=$(CUDA_ARCH) -lineinfo -Xcompiler=-Wall,-Wextra
LDLIBS    := -lm

ifeq ($(BUILD_TYPE),debug)
CFLAGS    += -O0 -g
NVCCFLAGS += -O0 -g -G
else ifeq ($(BUILD_TYPE),release)
CFLAGS    += -O3
NVCCFLAGS += -O3
else
$(error BUILD_TYPE must be either release or debug)
endif

CUDA_OBJECT := $(BUILD_DIR)/main.o
NAIVE_CUDA_OBJECT := $(BUILD_DIR)/gemm_naive.o
GEMM_OBJECT := $(BUILD_DIR)/gemm_cpu.o
MATRIX_OBJECT := $(BUILD_DIR)/matrix.o

GEMM_TEST_OBJECT   := $(BUILD_DIR)/test_gemm_cpu.o
MATRIX_TEST_OBJECT := $(BUILD_DIR)/test_matrix.o
CUDA_TEST_OBJECT   := $(BUILD_DIR)/test_gemm_cuda.o
BENCHMARK_OBJECT   := $(BUILD_DIR)/benchmark_gemm.o

GEMM_TEST_TARGET   := $(BUILD_DIR)/test_gemm_cpu
MATRIX_TEST_TARGET := $(BUILD_DIR)/test_matrix
CUDA_TEST_TARGET   := $(BUILD_DIR)/test_gemm_cuda
BENCHMARK_TARGET   := $(BUILD_DIR)/benchmark_gemm
CPU_TEST_TARGETS   := $(GEMM_TEST_TARGET) $(MATRIX_TEST_TARGET)

.PHONY: all run test test-cpu test-cuda benchmark debug sanitize clean

all: $(TARGET) $(CPU_TEST_TARGETS) $(CUDA_TEST_TARGET) $(BENCHMARK_TARGET)

$(TARGET): $(CUDA_OBJECT)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(CUDA_OBJECT): src/main.cu include/cuda_check.h
	@mkdir -p $(dir $@)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c $< -o $@

$(NAIVE_CUDA_OBJECT): src/gemm_naive.cu include/gemm_cuda.h
	@mkdir -p $(dir $@)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c $< -o $@

$(GEMM_OBJECT): src/gemm_cpu.c include/gemm.h
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(MATRIX_OBJECT): src/matrix.c include/matrix.h
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(GEMM_TEST_OBJECT): tests/test_gemm_cpu.c include/gemm.h include/matrix.h
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(MATRIX_TEST_OBJECT): tests/test_matrix.c include/matrix.h
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(CUDA_TEST_OBJECT): tests/test_gemm_cuda.cu include/cuda_check.h \
                    include/gemm.h include/gemm_cuda.h include/matrix.h
	@mkdir -p $(dir $@)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c $< -o $@

$(BENCHMARK_OBJECT): benchmarks/benchmark_gemm.cu include/cuda_check.h \
                     include/gemm_cuda.h include/matrix.h
	@mkdir -p $(dir $@)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c $< -o $@

$(GEMM_TEST_TARGET): $(GEMM_TEST_OBJECT) $(GEMM_OBJECT) $(MATRIX_OBJECT)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

$(MATRIX_TEST_TARGET): $(MATRIX_TEST_OBJECT) $(MATRIX_OBJECT)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

$(CUDA_TEST_TARGET): $(CUDA_TEST_OBJECT) $(NAIVE_CUDA_OBJECT) \
                     $(GEMM_OBJECT) $(MATRIX_OBJECT)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

$(BENCHMARK_TARGET): $(BENCHMARK_OBJECT) $(NAIVE_CUDA_OBJECT) $(MATRIX_OBJECT)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

run: $(TARGET)
	./$(TARGET)

test: test-cpu test-cuda

test-cpu: $(CPU_TEST_TARGETS)
	./$(GEMM_TEST_TARGET)
	./$(MATRIX_TEST_TARGET)

test-cuda: $(CUDA_TEST_TARGET)
	./$(CUDA_TEST_TARGET)

benchmark: $(BENCHMARK_TARGET)
	@./$(BENCHMARK_TARGET)


debug:
	$(MAKE) BUILD_TYPE=debug

sanitize: $(CUDA_TEST_TARGET)
	compute-sanitizer --tool memcheck ./$(CUDA_TEST_TARGET)
	compute-sanitizer --tool synccheck ./$(CUDA_TEST_TARGET)

clean:
	rm -rf build

