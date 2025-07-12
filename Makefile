# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2 -fopenmp
LDFLAGS = -lm -lpthread

# Sources and headers
C_SOURCES = $(wildcard matrix/*.c neural/*.c util/*.c socket/*.c)
PIPELINE_SOURCES = $(wildcard matrix/*.c neural/*.c util/*.c socket/*.c pipeline/*.c)
HEADERS = $(wildcard matrix/*.h neural/*.h util/*.h *.h socket/*.h pipeline/*.h)

# Object files
OBJ = $(C_SOURCES:.c=.o)
PIPELINE_OBJ = $(PIPELINE_SOURCES:.c=.o)

# Executable
TARGET = app
PIPELINE_TARGET = app_pipeline
PREDICT = predict
TRAIN_SINGLE = train_single
THREAD_TEST = test_thread_performance

# Default target
all: $(TARGET) $(PIPELINE_TARGET) $(TRAIN_SINGLE)

# Build the app
$(TARGET): train.c $(OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Build the pipeline app
$(PIPELINE_TARGET): train_pipeline.c $(PIPELINE_OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Build predict
$(PREDICT): predict.c $(OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Build single training
$(TRAIN_SINGLE): train_single.c $(OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Build thread performance test
$(THREAD_TEST): test_thread_performance.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Object file rule
%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

# Test thread performance
test-threads: $(THREAD_TEST)
	@echo "🧪 Running thread performance benchmark..."
	@echo "Limiting to 2 CPU cores (simulating pipeline environment)"
	taskset -c 0,1 ./$(THREAD_TEST)

# Clean target
clean:
	rm -f matrix/*.o neural/*.o util/*.o socket/*.o pipeline/*.o *.o $(TARGET) $(PIPELINE_TARGET) $(PREDICT) $(TRAIN_SINGLE) $(THREAD_TEST)

# Pipeline Parallelism Network Simulation Makefile

.PHONY: help build up down logs simulate-lan simulate-wan simulate-slow simulate-unstable reset-network monitor clean test-threads

help:
	@echo "Pipeline Parallelism Network Simulation Commands:"
	@echo ""
	@echo "Basic Docker operations:"
	@echo "  build              Build the Docker images"
	@echo "  up                 Start the pipeline containers"
	@echo "  down               Stop and remove containers"
	@echo "  logs               Show container logs"
	@echo ""
	@echo "Network Simulation:"
	@echo "  simulate-lan       Simulate LAN conditions (2ms latency, 100mbps)"
	@echo "  simulate-wan       Simulate WAN conditions (100ms latency, 10mbps)"
	@echo "  simulate-slow      Simulate slow network (200ms latency, 1mbps)"
	@echo "  simulate-unstable  Simulate unstable network (50ms latency, 5mbps, 3% loss)"
	@echo "  reset-network      Reset to default Docker networking"
	@echo ""
	@echo "Monitoring:"
	@echo "  monitor            Monitor network for 60 seconds"
	@echo "  monitor-long       Monitor network for 300 seconds"
	@echo ""
	@echo "Utilities:"
	@echo "  clean              Clean up monitoring results"
	@echo "  status             Show current network status"

build:
	docker-compose -f docker-compose-pipeline.yml build

up:
	docker-compose -f docker-compose-pipeline.yml up -d
	@echo "Waiting for containers to start..."
	@sleep 10
	@echo "Pipeline containers are running!"

down:
	docker-compose -f docker-compose-pipeline.yml down

logs:
	docker-compose -f docker-compose-pipeline.yml logs -f

simulate-lan:
	@echo "Setting up LAN simulation..."
	@chmod +x scripts/setup-network-simulation.sh
	@./scripts/setup-network-simulation.sh lan

simulate-wan:
	@echo "Setting up WAN simulation..."
	@chmod +x scripts/setup-network-simulation.sh
	@./scripts/setup-network-simulation.sh wan

simulate-slow:
	@echo "Setting up slow network simulation..."
	@chmod +x scripts/setup-network-simulation.sh
	@./scripts/setup-network-simulation.sh slow

simulate-unstable:
	@echo "Setting up unstable network simulation..."
	@chmod +x scripts/setup-network-simulation.sh
	@./scripts/setup-network-simulation.sh unstable

reset-network:
	@echo "Resetting network to defaults..."
	@chmod +x scripts/reset-network.sh
	@./scripts/reset-network.sh

monitor:
	@echo "Starting 60-second network monitoring..."
	@chmod +x scripts/monitor-network.sh
	@./scripts/monitor-network.sh 60

monitor-long:
	@echo "Starting 5-minute network monitoring..."
	@chmod +x scripts/monitor-network.sh
	@./scripts/monitor-network.sh 300

status:
	@echo "=== Container Status ==="
	@docker-compose -f docker-compose-pipeline.yml ps
	@echo ""
	@echo "=== Network Configuration ==="
	@docker exec pipeline-parallelism-improve-stage1-1 tc qdisc show dev eth0 2>/dev/null || echo "Stage1: Default networking"
	@docker exec pipeline-parallelism-improve-stage2-1 tc qdisc show dev eth0 2>/dev/null || echo "Stage2: Default networking" 
	@docker exec pipeline-parallelism-improve-coordinator-1 tc qdisc show dev eth0 2>/dev/null || echo "Coordinator: Default networking"

clean:
	@echo "Cleaning up monitoring results..."
	@rm -rf monitoring_results/
	@echo "Monitoring results cleaned!"

# Complete workflow examples
test-lan: up simulate-lan monitor
	@echo "LAN simulation test completed!"

test-wan: up simulate-wan monitor
	@echo "WAN simulation test completed!"

test-slow: up simulate-slow monitor
	@echo "Slow network simulation test completed!"
