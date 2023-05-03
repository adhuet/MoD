CC = g++
NVCC = nvcc
SHELL = bash

INCLUDE = -Iinclude

CPU_VERSION = 1.0
GPU_VERSION = 1.1
OCV_VERSION = 1.0

CFLAGS = -Wall -Werror -Wextra -O3 -std=c++17 -g
# CFLAGS = -Wall -Werror -Wextra -O3 -std=c++17 -fsanitize=address -g
# CFLAGS += $(shell pkg-config --cflags opencv4)
CV_FLAGS = $(shell pkg-config --cflags opencv4)
LD_LIBS = $(shell pkg-config --libs opencv4)
# LD_LIBS = $(shell pkg-config --libs opencv4) -fsanitize=address


# CPU IMPLEMENTATION
CPU_SRC = $(wildcard $(addsuffix /*.cpp, src/cpu))
CPU_SRC := $(filter-out src/cpu/main.cpp, $(CPU_SRC))
CPU_OBJS = $(CPU_SRC:.cpp=.o)
CPU_BIN = mod_cpu

# UTILS FILES
UTILS_SRC = $(wildcard $(addsuffix /*.cpp, src/utils))
UTILS_OBJS = $(UTILS_SRC:.cpp=.o)

# CUDA IMPLEMENTATION
CUDA_FLAGS = -O3 -arch=sm_50 -Xcudafe --diag_suppress=611
CUDA_SRC = $(wildcard $(addsuffix /*.cu, src/gpu))
CUDA_SRC := $(filter-out src/gpu/main.cu, $(CUDA_SRC))
CUDA_OBJS = $(CUDA_SRC:.cu=.o)
CUDA_BIN = mod_gpu

# TEST FILES
TEST_SRC = $(wildcard $(addsuffix /*.cpp, tests))
TEST_OBJS = $(TEST_SRC:.cpp=.o)
TEST_BIN = testsuite
GPUTESTS_SRC = src/gpu/gpuTests/gpu_tests.cu
GPUTESTS_OBJS = $(GPUTESTS_SRC:.cu=.o)
GPUTESTS_BIN = gputestsuite

# ALL OBJS
OBJS = $(CPU_OBJS) $(UTILS_OBJS) $(CUDA_OBJS) $(TEST_OBJS) $(GPUTESTS_OBJS)

# BINARIES
BINS = $(CPU_BIN) $(CUDA_BIN) $(TEST_BIN)

INPUT_FILE = data/pigeon.mp4

all: $(CPU_BIN) $(CUDA_BIN)

$(CPU_BIN): $(CPU_OBJS) $(UTILS_OBJS) src/cpu/main.o
	$(CC) -o $@ $^ $(LD_LIBS)

$(CUDA_BIN): $(CUDA_OBJS) $(UTILS_OBJS) src/gpu/main.o
	$(NVCC) -o $@ $^ $(LD_LIBS)

$(TEST_BIN): $(UTILS_OBJS) $(TEST_OBJS) $(CPU_OBJS)
	$(CC) -o $@ $^ -lcriterion $(LD_LIBS)

check: $(TEST_BIN)
	./testsuite -j4

$(GPUTESTS_BIN): $(GPUTESTS_OBJS) $(CUDA_OBJS) $(UTILS_OBJS) src/cpu/blur.o
	$(NVCC) -o $@ $^ -lcriterion $(LD_LIBS)

gpucheck: $(GPUTESTS_BIN)
	./gputestsuite

# run: $(CPU_BIN)
# 	./$(CPU_BIN) $(INPUT_FILE)

run: $(CUDA_BIN)
	./$(CUDA_BIN) $(INPUT_FILE)

%.o: %.cpp
	$(CC) -c $(CFLAGS) $(CV_FLAGS) $(INCLUDE) -o $@ $<

%.o: %.cu
	$(NVCC) -c $(CUDA_FLAGS) $(CV_FLAGS) $(INCLUDE) -o $@ $<

benchsuite: CUDA_FLAGS += -D_CPU_VERSION=$(CPU_VERSION) -D_GPU_VERSION=$(GPU_VERSION) -D_OCV_VERSION=$(OCV_VERSION)
benchsuite: src/benchmark.o $(CPU_OBJS) $(UTILS_OBJS) $(CUDA_OBJS)
	$(NVCC) -o $@ $^ $(LD_LIBS)

bench: benchsuite
	./benchsuite $(INPUT_FILE)

report: benchsuite
	@echo "Generating report..."
	@touch 'reports/report_OVCv$(OCV_VERSION)_CPUv$(CPU_VERSION)_GPUv$(GPU_VERSION).txt'
	@./benchsuite $(INPUT_FILE) > reports/report_OVCv$(OCV_VERSION)_CPUv$(CPU_VERSION)_GPUv$(GPU_VERSION).txt
	@echo Done
clean:
	$(RM) $(BINS) $(OBJS) src/cpu/main.o src/gpu/main.o src/benchmark.o gputestsuite benchsuite

.PHONY: all clean run bench check gpucheck