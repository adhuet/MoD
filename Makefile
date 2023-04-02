CC = g++
NVCC = nvcc
SHELL = bash

INCLUDE = -Iinclude

CFLAGS = -Wall -Werror -Wextra -O3
CFLAGS += $(shell pkg-config --cflags opencv4)

LD_LIBS = $(shell pkg-config --libs opencv4)

CPU_SRC = $(wildcard $(addsuffix /*.cpp, src/cpu))
CPU_OBJS = $(CPU_SRC:.cpp=.o)

CPU_BIN = mod_cpu

BINS = $(CPU_BIN)
OBJS = $(CPU_OBJS)

all: $(CPU_BIN)

$(CPU_BIN): $(CPU_OBJS)
	$(CC) -o $@ $^ $(LD_LIBS)

%.o: %.cpp
	$(CC) -c $(CFLAGS) $(INCLUDE) -o $@ $<

clean:
	$(RM) $(BINS) $(OBJS)

.PHONY: all clean