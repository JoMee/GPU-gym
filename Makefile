CC   ?= cc
NVCC ?= nvcc

CUDA_ARCH  ?= native
BUILD_TYPE ?= release

BUILD_DIR := build/$(BUILD_TYPE)
TARGET    := $(BUILD_DIR)/gemm_lab

CPPFLAGS  := -Iinclude
CFLAGS    := -std=c17 -Wall -Wextra -Wpedantic
NVCCFLAGS := -arch=$(CUDA_ARCH) -lineinfo -Xcompiler=-Wall,-Wextra

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
GEMM_OBJECT := $(BUILD_DIR)/gemm_cpu.o
MATRIX_OBJECT := $(BUILD_DIR)/matrix.o

GEMM_TEST_OBJECT   := $(BUILD_DIR)/test_gemm_cpu.o
MATRIX_TEST_OBJECT := $(BUILD_DIR)/test_matrix.o

GEMM_TEST_TARGET   := $(BUILD_DIR)/test_gemm_cpu
MATRIX_TEST_TARGET := $(BUILD_DIR)/test_matrix
TEST_TARGETS       := $(GEMM_TEST_TARGET) $(MATRIX_TEST_TARGET)

.PHONY: all run test debug sanitize clean

all: $(TARGET) $(TEST_TARGETS)

$(TARGET): $(CUDA_OBJECT)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(CUDA_OBJECT): src/main.cu include/cuda_check.h
	@mkdir -p $(dir $@)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c $< -o $@

$(GEMM_OBJECT): src/gemm_cpu.c include/gemm.h
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(MATRIX_OBJECT): src/matrix.c include/matrix.h
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(GEMM_TEST_OBJECT): tests/test_gemm_cpu.c include/gemm.h
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(MATRIX_TEST_OBJECT): tests/test_matrix.c include/matrix.h
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(GEMM_TEST_TARGET): $(GEMM_TEST_OBJECT) $(GEMM_OBJECT)
	$(CC) $(CFLAGS) $^ -o $@

$(MATRIX_TEST_TARGET): $(MATRIX_TEST_OBJECT) $(MATRIX_OBJECT)
	$(CC) $(CFLAGS) $^ -o $@

run: $(TARGET)
	./$(TARGET)

test: $(TEST_TARGETS)
	./$(GEMM_TEST_TARGET)
	./$(MATRIX_TEST_TARGET)


debug:
	$(MAKE) BUILD_TYPE=debug

sanitize: $(TARGET)
	compute-sanitizer --tool memcheck ./$(TARGET)

clean:
	rm -rf build

