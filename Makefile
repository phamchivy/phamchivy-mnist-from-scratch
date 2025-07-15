# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2 -fopenmp
LDFLAGS = -lm -lpthread

# Source directories
MATRIX_DIR = matrix
NEURAL_DIR = neural
UTIL_DIR = util
SOCKET_DIR = socket
PIPELINE_DIR = pipeline

# Sources and headers
C_SOURCES = $(wildcard $(MATRIX_DIR)/*.c $(NEURAL_DIR)/*.c $(UTIL_DIR)/*.c $(SOCKET_DIR)/*.c)
PIPELINE_SOURCES = $(wildcard $(MATRIX_DIR)/*.c $(NEURAL_DIR)/*.c $(UTIL_DIR)/*.c $(SOCKET_DIR)/*.c $(PIPELINE_DIR)/*.c)

# Hybrid system sources
HYBRID_SOURCES = $(PIPELINE_SOURCES) \
	$(PIPELINE_DIR)/easgd_core.c \
	$(PIPELINE_DIR)/hybrid_worker.c \
	$(PIPELINE_DIR)/hybrid_coordinator.c \
	$(PIPELINE_DIR)/weight_management.c \
	$(PIPELINE_DIR)/hybrid_metrics.c

HEADERS = $(wildcard $(MATRIX_DIR)/*.h $(NEURAL_DIR)/*.h $(UTIL_DIR)/*.h $(SOCKET_DIR)/*.h $(PIPELINE_DIR)/*.h)

# Object files
OBJ = $(C_SOURCES:.c=.o)
PIPELINE_OBJ = $(PIPELINE_SOURCES:.c=.o)
HYBRID_OBJ = $(HYBRID_SOURCES:.c=.o)

# Executables
PIPELINE_TARGET = app_pipeline
HYBRID_TARGET = app_hybrid
PREDICT = predict
TRAIN_SINGLE = train_single
THREAD_TEST = test_thread_performance

# Default target
all: $(HYBRID_TARGET) $(TRAIN_SINGLE)

# # Build the original pipeline app
# $(PIPELINE_TARGET): train_pipeline.c $(PIPELINE_OBJ)
# 	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Build the hybrid pipeline-EASGD app
$(HYBRID_TARGET): $(HYBRID_OBJ)
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

# Test hybrid system locally
test-hybrid-local: $(HYBRID_TARGET)
	@echo "🧪 Testing hybrid system locally..."
	@echo "Starting coordinator in background..."
	@./$(HYBRID_TARGET) coordinator 0.5 &
	@sleep 2
	@echo "Starting Stage2 worker in background..."
	@./$(HYBRID_TARGET) stage2 0.3 &
	@sleep 2
	@echo "Starting Stage1 workers..."
	@./$(HYBRID_TARGET) stage1 1 0 30000 0.2 &
	@./$(HYBRID_TARGET) stage1 2 30000 60000 0.2 &
	@echo "Hybrid system test started. Use 'pkill app_hybrid' to stop."

# Clean target
clean:
	rm -f $(MATRIX_DIR)/*.o $(NEURAL_DIR)/*.o $(UTIL_DIR)/*.o $(SOCKET_DIR)/*.o $(PIPELINE_DIR)/*.o *.o 
	rm -f $(PIPELINE_TARGET) $(HYBRID_TARGET) $(PREDICT) $(TRAIN_SINGLE) $(THREAD_TEST)

# Pipeline Parallelism Network Simulation Makefile

.PHONY: help build build-hybrid up up-hybrid down logs simulate-lan simulate-wan simulate-slow simulate-unstable reset-network monitor clean test-threads test-hybrid-local

help:
	@echo "Pipeline Parallelism + EASGD Network Simulation Commands:"
	@echo ""
	@echo "Build targets:"
	@echo "  build              Build the original pipeline Docker images"
	@echo "  build-hybrid       Build the hybrid pipeline-EASGD Docker images"
	@echo "  test-hybrid-local  Test hybrid system locally (single machine)"
	@echo ""
	@echo "Docker operations (Original Pipeline):"
	@echo "  up                 Start the original pipeline containers"
	@echo "  down               Stop and remove containers"
	@echo "  logs               Show container logs"
	@echo ""
	@echo "Docker operations (Hybrid Pipeline-EASGD):"
	@echo "  up-hybrid          Start the hybrid pipeline-EASGD containers"
	@echo "  down-hybrid        Stop and remove hybrid containers"
	@echo "  logs-hybrid        Show hybrid container logs"
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
	@echo "Research workflows:"
	@echo "  experiment-lan     Run complete LAN experiment (hybrid + monitoring)"
	@echo "  experiment-wan     Run complete WAN experiment (hybrid + monitoring)"
	@echo "  experiment-local   Run complete LOCAL experiment (hybrid + monitoring)"
	@echo ""
	@echo "Utilities:"
	@echo "  clean              Clean up monitoring results"
	@echo "  status             Show current network status"

# Original pipeline Docker operations
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

# Hybrid pipeline Docker operations
build-hybrid:
	docker-compose -f docker-compose-hybrid.yml build

up-hybrid:
	@echo "Starting hybrid pipeline-EASGD system..."
	docker-compose -f docker-compose-hybrid.yml up -d
	@echo "Waiting for containers to start..."
	@sleep 10
	@echo "Hybrid pipeline-EASGD containers are running!"
	@echo ""
	@echo "Container status:"
	@docker-compose -f docker-compose-hybrid.yml ps
	@echo ""
	@echo "Use 'make logs-hybrid' to monitor training progress"

down-hybrid:
	docker-compose -f docker-compose-hybrid.yml down

logs-hybrid:
	docker-compose -f docker-compose-hybrid.yml logs -f

# Network simulation
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

# Research experiment workflows
experiment-local: build-hybrid
	@echo "=== Running LOCAL Network Experiment ==="
	@echo "1. Starting hybrid system..."
	@make up-hybrid
	@sleep 5
	@echo "2. Monitoring for 180 seconds..."
	@make monitor DURATION=180
	@echo "3. Collecting results..."
	@docker-compose -f docker-compose-hybrid.yml logs > results/local_experiment_logs.txt
	@echo "4. Stopping containers..."
	@make down-hybrid
	@echo "LOCAL experiment completed. Results in results/ directory."

experiment-lan: build-hybrid simulate-lan
	@echo "=== Running LAN Network Experiment ==="
	@echo "1. Starting hybrid system with LAN simulation..."
	@NETWORK_TYPE=LAN make up-hybrid
	@sleep 5
	@echo "2. Monitoring for 240 seconds..."
	@make monitor DURATION=240
	@echo "3. Collecting results..."
	@mkdir -p results
	@docker-compose -f docker-compose-hybrid.yml logs > results/lan_experiment_logs.txt
	@echo "4. Stopping containers and resetting network..."
	@make down-hybrid
	@make reset-network
	@echo "LAN experiment completed. Results in results/ directory."

experiment-wan: build-hybrid simulate-wan
	@echo "=== Running WAN Network Experiment ==="
	@echo "1. Starting hybrid system with WAN simulation..."
	@NETWORK_TYPE=WAN make up-hybrid
	@sleep 5
	@echo "2. Monitoring for 300 seconds..."
	@make monitor DURATION=300
	@echo "3. Collecting results..."
	@mkdir -p results
	@docker-compose -f docker-compose-hybrid.yml logs > results/wan_experiment_logs.txt
	@echo "4. Stopping containers and resetting network..."
	@make down-hybrid
	@make reset-network
	@echo "WAN experiment completed. Results in results/ directory."

# Complete research workflow
run-all-experiments: clean
	@echo "=== Running Complete Research Experiment Suite ==="
	@mkdir -p results
	@echo "Experiment suite started at $(shell date)" > results/experiment_log.txt
	@echo ""
	@echo "Running LOCAL experiment..."
	@make experiment-local
	@echo "LOCAL experiment completed at $(shell date)" >> results/experiment_log.txt
	@sleep 30
	@echo ""
	@echo "Running LAN experiment..."
	@make experiment-lan  
	@echo "LAN experiment completed at $(shell date)" >> results/experiment_log.txt
	@sleep 30
	@echo ""
	@echo "Running WAN experiment..."
	@make experiment-wan
	@echo "WAN experiment completed at $(shell date)" >> results/experiment_log.txt
	@echo ""
	@echo "=== All experiments completed! ==="
	@echo "Results available in results/ directory:"
	@ls -la results/

status:
	@echo "=== Container Status ==="
	@if [ -f docker-compose-hybrid.yml ]; then \
		docker-compose -f docker-compose-hybrid.yml ps; \
	else \
		docker-compose -f docker-compose-pipeline.yml ps; \
	fi
	@echo ""
	@echo "=== Network Configuration ==="
	@docker exec pipeline-parallelism-improve-stage1-1-1 tc qdisc show dev eth0 2>/dev/null || echo "Stage1-1: Default networking"
	@docker exec pipeline-parallelism-improve-stage1-2-1 tc qdisc show dev eth0 2>/dev/null || echo "Stage1-2: Default networking"
	@docker exec pipeline-parallelism-improve-stage2-1 tc qdisc show dev eth0 2>/dev/null || echo "Stage2: Default networking" 
	@docker exec pipeline-parallelism-improve-coordinator-1 tc qdisc show dev eth0 2>/dev/null || echo "Coordinator: Default networking"

clean:
	@echo "Cleaning up monitoring results and build artifacts..."
	@rm -rf monitoring_results/
	@rm -rf results/
	@rm -f *.csv *.log
	@echo "Cleanup completed!"

# Development and debugging targets
debug-hybrid: $(HYBRID_TARGET)
	@echo "🔍 Running hybrid system in debug mode..."
	@echo "Building with debug symbols..."
	@$(CC) $(CFLAGS) -g -DDEBUG $(PIPELINE_DIR)/hybrid_main.c $(HYBRID_OBJ) -o $(HYBRID_TARGET)_debug $(LDFLAGS)
	@echo "Debug build complete. Use gdb ./$(HYBRID_TARGET)_debug for debugging."

check-dependencies:
	@echo "🔍 Checking system dependencies..."
	@echo -n "GCC: "; gcc --version | head -n1
	@echo -n "Docker: "; docker --version
	@echo -n "Docker Compose: "; docker-compose --version
	@echo -n "Make: "; make --version | head -n1
	@echo -n "OpenMP support: "; echo "#include <omp.h>" | gcc -fopenmp -x c - -o /dev/null 2>/dev/null && echo "OK" || echo "NOT FOUND"
	@echo "Dependencies check completed."

# Performance profiling targets
profile-hybrid: $(HYBRID_TARGET)
	@echo "🔬 Building hybrid system with profiling support..."
	@$(CC) $(CFLAGS) -pg $(PIPELINE_DIR)/hybrid_main.c $(HYBRID_OBJ) -o $(HYBRID_TARGET)_profile $(LDFLAGS)
	@echo "Profiling build complete. Run system and use 'gprof' for analysis."

# Documentation generation
docs:
	@echo "📚 Generating documentation..."
	@mkdir -p docs
	@echo "# Hybrid Pipeline-EASGD Documentation" > docs/README.md
	@echo "" >> docs/README.md
	@echo "## System Architecture" >> docs/README.md
	@echo "- Stage1-1: Data partition [0:30000] with α₁ parameter" >> docs/README.md
	@echo "- Stage1-2: Data partition [30000:60000] with α₁ parameter" >> docs/README.md
	@echo "- Stage2-1: Multiplexed processing with α₂ parameter" >> docs/README.md
	@echo "- Coordinator: EASGD Parameter Server with β parameter" >> docs/README.md
	@echo "" >> docs/README.md
	@echo "## Usage Examples" >> docs/README.md
	@echo "\`\`\`bash" >> docs/README.md
	@echo "# Local testing" >> docs/README.md
	@echo "make test-hybrid-local" >> docs/README.md
	@echo "" >> docs/README.md
	@echo "# Full experiments" >> docs/README.md
	@echo "make experiment-local" >> docs/README.md
	@echo "make experiment-lan" >> docs/README.md
	@echo "make experiment-wan" >> docs/README.md
	@echo "\`\`\`" >> docs/README.md
	@echo "Documentation generated in docs/"

# Complete workflow examples
test-lan: up-hybrid simulate-lan monitor
	@echo "LAN simulation test completed!"

test-wan: up-hybrid simulate-wan monitor
	@echo "WAN simulation test completed!"

test-slow: up-hybrid simulate-slow monitor
	@echo "Slow network simulation test completed!"

# Quick validation
validate-hybrid: $(HYBRID_TARGET)
	@echo "✅ Validating hybrid system build..."
	@./$(HYBRID_TARGET) --help > /dev/null && echo "✅ Help command works" || echo "❌ Help command failed"
	@echo "✅ Hybrid system validation completed"

# Development targets
dev-build: $(HYBRID_TARGET)
	@echo "🔧 Building development version..."
	@echo "✅ Ready for fast iteration with volume mounting"

dev-run: dev-build
	@echo "🚀 Running development mode..."
	@echo "Use: ./launch_dev.sh docker local"
	@echo "Or: ./launch_dev.sh local local"

dev-restart:
	@echo "🔄 Restarting development containers..."
	docker-compose -f docker-compose-dev.yml restart

dev-logs:
	@echo "📋 Showing development logs..."
	docker-compose -f docker-compose-dev.yml logs -f

dev-clean:
	@echo "🧹 Cleaning development environment..."
	docker-compose -f docker-compose-dev.yml down
	docker system prune -f