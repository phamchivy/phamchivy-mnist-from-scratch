#!/bin/bash

# Network Simulation Setup for Pipeline Parallelism Testing
# Usage: ./setup-network-simulation.sh [scenario]

SCENARIO=${1:-"default"}

echo "Setting up network simulation - Scenario: $SCENARIO"

case $SCENARIO in
    "lan")
        echo "Configuring LAN-like network conditions..."
        DELAY="2ms"
        BANDWIDTH="100mbit"
        LOSS="0.01%"
        JITTER="1ms"
        ;;
    "wan")
        echo "Configuring WAN-like network conditions..."
        DELAY="100ms"
        BANDWIDTH="10mbit"
        LOSS="0.5%"
        JITTER="10ms"
        ;;
    "slow")
        echo "Configuring slow network conditions..."
        DELAY="200ms"
        BANDWIDTH="1mbit"
        LOSS="2%"
        JITTER="20ms"
        ;;
    "unstable")
        echo "Configuring unstable network conditions..."
        DELAY="50ms"
        BANDWIDTH="5mbit"
        LOSS="3%"
        JITTER="25ms"
        ;;
    "default"|*)
        echo "Configuring default simulation..."
        DELAY="50ms"
        BANDWIDTH="10mbit"
        LOSS="1%"
        JITTER="5ms"
        ;;
esac

# Apply network conditions to all containers
CONTAINERS=("pipeline-parallelism-improve-stage1-1" "pipeline-parallelism-improve-stage2-1" "pipeline-parallelism-improve-coordinator-1")

for container in "${CONTAINERS[@]}"; do
    echo "Applying network simulation to $container..."
    
    # Remove existing rules
    docker exec -it $container sh -c "tc qdisc del dev eth0 root 2>/dev/null || true"
    
    # Add new network conditions
    docker exec -it $container sh -c "
        tc qdisc add dev eth0 root handle 1: netem delay $DELAY $JITTER loss $LOSS &&
        tc qdisc add dev eth0 parent 1:1 handle 2: tbf rate $BANDWIDTH burst 32kbit latency 300ms
    "
    
    if [ $? -eq 0 ]; then
        echo "✓ Network simulation applied to $container"
    else
        echo "✗ Failed to apply network simulation to $container"
    fi
done

echo ""
echo "Network Simulation Summary:"
echo "- Latency: $DELAY ± $JITTER"
echo "- Bandwidth: $BANDWIDTH"
echo "- Packet Loss: $LOSS"
echo ""
echo "To verify settings, run:"
echo "docker exec -it pipeline-parallelism-improve-stage1-1 tc qdisc show dev eth0" 