NVCC ?= nvcc

CUDA_ARCH  ?= native
BUILD_TYPE ?= release

BUILD_DIR := build/$(BUILD_TYPE)
TARGET    := $(BUILD_DIR)/gemm_lab

CPPFLAGS  := -Iinclude
NVCCFLAGS := -arch=$(CUDA_ARCH) -lineinfo -Xcompiler=-Wall,-Wextra

ifeq ($(BUILD_TYPE),debug)
NVCCFLAGS += -O0 -g -G
else ifeq ($(BUILD_TYPE),release)
NVCCFLAGS += -O3
else
$(error BUILD_TYPE must be either release or debug)
endif

SOURCES := src/main.cu
OBJECTS := $(SOURCES:src/%.cu=$(BUILD_DIR)/%.o)

.PHONY: all run debug sanitize clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(BUILD_DIR)/%.o: src/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

debug:
	$(MAKE) BUILD_TYPE=debug

sanitize: $(TARGET)
	compute-sanitizer --tool memcheck ./$(TARGET)

clean:
	rm -rf $(BUILD_DIR)

