CC = g++
NVCC = nvcc
SHELL = bash

INCLUDE = -Iinclude

CFLAGS = -c -Wall -Werre -Wextra -O3
CLFAGS += $(shell v='$(pkg-config --libs --cflags opencv)'; echo "$${v%.*}")


CPU_SRC = $(wildcard $(addsuffix /*.cpp, src/cpu))
CPU_OBJS = $(CPU_SRC:.cpp=.o)

CPU_BIN = mod_cpu

BINS = $(CPU_BIN)
OBJS = $(CPU_OBJS)

all: $(CPU_BIN)

$(CPU_BIN): $(CPU_OBJS)
	$(CC) -o $@ $^

%.o: %.cpp
	$(CC) -c $(CLFAGS) $(INCLUDE) -o $@ $<

clean:
	$(RM) $(BINS) $(OBJS)

.PHONY: all clean