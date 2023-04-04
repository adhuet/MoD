CC = g++
NVCC = nvcc
SHELL = bash

INCLUDE = -Iinclude

CFLAGS = -Wall -Werror -Wextra -O3
# CFLAGS = -Wall -Werror -Wextra -O3 -fsanitize=address -g
CFLAGS += $(shell pkg-config --cflags opencv4) 

LD_LIBS = $(shell pkg-config --libs opencv4)
# LD_LIBS = $(shell pkg-config --libs opencv4) -fsanitize=address


# CPU IMPLEMENTATION
CPU_SRC = $(wildcard $(addsuffix /*.cpp, src/cpu))
CPU_OBJS = $(CPU_SRC:.cpp=.o)
CPU_BIN = mod_cpu

# UTILS FILES
UTILS_SRC = $(wildcard $(addsuffix /*.cpp, src/utils))
UTILS_OBJS = $(UTILS_SRC:.cpp=.o)

# CUDA IMPLEMENTATION
CUDA_FLAGS = -O3 -arch=sm_50 $(shell pkg-config --cflags opencv4) 
CUDA_SRC = $(wildcard $(addsuffix /*.cu, src/gpu))
CUDA_OBJS = $(CUDA_SRC:.cu=.o)
CUDA_BIN = mod_gpu

# ALL OBJS
OBJS = $(CPU_OBJS) $(UTILS_OBJS) $(CUDA_OBJS)

# BINARIES
BINS = $(CPU_BIN) $(CUDA_BIN)

INPUT_FILE = data/pigeon.mp4

all: $(CPU_BIN)

$(CPU_BIN): $(CPU_OBJS) $(UTILS_OBJS)
	$(CC) -o $@ $^ $(LD_LIBS)

$(CUDA_BIN): $(CUDA_OBJS) $(UTILS_OBJS)
	$(NVCC) -o $@ $^ $(LD_LIBS)

run: $(CPU_BIN)
	./$(CPU_BIN) $(INPUT_FILE)

%.o: %.cpp
	$(CC) -c $(CFLAGS) $(INCLUDE) -o $@ $<

%.o: %.cu
	$(NVCC) -c $(CUDA_FLAGS) $(INCLUDE) -o $@ $<

clean:
	$(RM) $(BINS) $(OBJS)

.PHONY: all clean run