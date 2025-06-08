#!/bin/bash

# Reset Network Simulation Script
# Usage: ./reset-network.sh

echo "Resetting network simulation to default Docker behavior..."

CONTAINERS=("pipeline-parallelism-improve-stage1-1" "pipeline-parallelism-improve-stage2-1" "pipeline-parallelism-improve-coordinator-1")

for container in "${CONTAINERS[@]}"; do
    echo "Resetting network for $container..."
    
    # Remove all traffic control rules
    docker exec -it $container sh -c "tc qdisc del dev eth0 root 2>/dev/null || true"
    
    if [ $? -eq 0 ]; then
        echo "✓ Network reset for $container"
    else
        echo "✗ Failed to reset network for $container (might already be clean)"
    fi
done

echo ""
echo "Network simulation reset completed!"
echo "All containers now use default Docker networking behavior."
echo ""
echo "To verify, run:"
echo "docker exec -it pipeline-parallelism-improve-stage1-1 tc qdisc show dev eth0" 