CC = gcc
CFLAGS = -std=c99 -Wall -Wextra -Wpedantic -g -fopenmp -D_POSIX_C_SOURCE=199309L
LDFLAGS = -lm -lpthread -lgomp

# Source directories
MATRIX_DIR = matrix
NEURAL_DIR = neural
SOCKET_DIR = socket
UTIL_DIR = util
APPS_DIR = apps

# Source files
MATRIX_SRCS = $(wildcard $(MATRIX_DIR)/*.c)
NEURAL_SRCS = $(wildcard $(NEURAL_DIR)/*.c)
SOCKET_SRCS = $(wildcard $(SOCKET_DIR)/*.c)
UTIL_SRCS = $(wildcard $(UTIL_DIR)/*.c)

# Object files
MATRIX_OBJS = $(MATRIX_SRCS:.c=.o)
NEURAL_OBJS = $(NEURAL_SRCS:.c=.o)
SOCKET_OBJS = $(SOCKET_SRCS:.c=.o)
UTIL_OBJS = $(UTIL_SRCS:.c=.o)

# Common objects
COMMON_OBJS = $(MATRIX_OBJS) $(NEURAL_OBJS) $(SOCKET_OBJS) $(UTIL_OBJS)

# Targets
all: setup parameter_server worker app

# Parameter server
parameter_server: $(COMMON_OBJS) $(APPS_DIR)/parameter_server.o
	$(CC) $^ -o $@ $(LDFLAGS)

# Worker
worker: $(COMMON_OBJS) $(APPS_DIR)/worker.o  
	$(CC) $^ -o $@ $(LDFLAGS)

# THÊM VÀO CUỐI FILE Makefile

# Hybrid Pipeline targets
worker_stage1: apps/worker_stage1.o $(COMMON_OBJS)
	$(CC) $^ -o $@ $(LDFLAGS)

worker_stage2: apps/worker_stage2.o $(COMMON_OBJS)
	$(CC) $^ -o $@ $(LDFLAGS)

socket/pipeline_socket.o: socket/pipeline_socket.c socket/pipeline_socket.h
	$(CC) -c socket/pipeline_socket.c -o socket/pipeline_socket.o $(CFLAGS)

neural/pipeline_utils.o: neural/pipeline_utils.c neural/pipeline_utils.h
	$(CC) -c neural/pipeline_utils.c -o neural/pipeline_utils.o $(CFLAGS)

# Build all hybrid components
hybrid: worker_stage1 worker_stage2 server

# Legacy app (for backward compatibility)
app: $(COMMON_OBJS) train.o
	$(CC) $^ -o $@ $(LDFLAGS)

# Server target alias
server: parameter_server

# Object file compilation
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean
clean:
	rm -f $(COMMON_OBJS) $(APPS_DIR)/*.o train.o
	rm -f parameter_server worker app

# Clean hybrid targets
clean_hybrid:
	rm -f worker_stage1 worker_stage2 socket/pipeline_socket.o neural/pipeline_utils.o

# Create results directories
setup:
	mkdir -p results/server_logs results/worker1_results results/worker2_results

.PHONY: all server worker clean setup

.PHONY: worker_stage1 worker_stage2 clean_hybrid
