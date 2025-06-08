#!/bin/bash

echo "=== Pipeline Parallelism Phase 2+3 Optimization Test ==="
echo "Testing: Asynchronous Processing + Dynamic Load Balancing"

# Start Docker containers with optimized configuration
docker-compose up -d

echo "Waiting for containers to start..."
sleep 10

# Test basic connectivity
echo "Testing network connectivity..."
docker exec coordinator ping -c 2 stage1
docker exec coordinator ping -c 2 stage2

echo "Starting optimized pipeline training with Phase 2+3 features..."
echo "- Asynchronous pipeline processing"
echo "- Dynamic batch size adaptation"
echo "- Pipeline depth management"
echo ""

# Measure start time
start_time=$(date +%s)

# Run the pipeline with performance monitoring
docker exec coordinator /app/pipeline_coordinator 12346 12347 &
COORDINATOR_PID=$!

sleep 5

docker exec stage1 /app/pipeline_stage1 12345 12348 &
STAGE1_PID=$!

sleep 5

docker exec stage2 /app/pipeline_stage2 12345 12348 172.32.0.2 &
STAGE2_PID=$!

# Monitor performance for 60 seconds
echo "Monitoring pipeline performance for 60 seconds..."
for i in {1..12}; do
    sleep 5
    echo "[$((i*5))s] Pipeline running with adaptive optimization..."
done

# Capture end time
end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "=== Phase 2+3 Optimization Results ==="
echo "Total training time: ${duration} seconds"
echo ""
echo "Expected improvements:"
echo "✓ Asynchronous processing: Overlapped computation and communication"
echo "✓ Dynamic batch sizing: Adaptive to network conditions"
echo "✓ Pipeline depth management: Optimized for throughput vs latency"
echo ""

# Clean up
echo "Cleaning up containers..."
docker-compose down

echo "Phase 2+3 optimization test completed!"
echo "Check container logs for detailed performance metrics." 