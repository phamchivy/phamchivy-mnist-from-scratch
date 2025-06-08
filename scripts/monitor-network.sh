#!/bin/bash

# Network Monitoring Script for Pipeline Parallelism
# Usage: ./monitor-network.sh [duration_seconds]

DURATION=${1:-60}
OUTPUT_DIR="./monitoring_results"

echo "Starting network monitoring for $DURATION seconds..."
mkdir -p $OUTPUT_DIR

# Test connectivity and latency between stages
test_connectivity() {
    echo "Testing connectivity between pipeline stages..."
    
    # Test stage1 -> stage2
    echo "=== Stage1 -> Stage2 ===" >> $OUTPUT_DIR/connectivity.log
    docker exec pipeline-parallelism-improve-stage1-1 ping -c 10 172.32.0.3 >> $OUTPUT_DIR/connectivity.log 2>&1
    
    # Test stage2 -> coordinator
    echo "=== Stage2 -> Coordinator ===" >> $OUTPUT_DIR/connectivity.log
    docker exec pipeline-parallelism-improve-stage2-1 ping -c 10 172.32.0.4 >> $OUTPUT_DIR/connectivity.log 2>&1
    
    # Test stage1 -> coordinator
    echo "=== Stage1 -> Coordinator ===" >> $OUTPUT_DIR/connectivity.log
    docker exec pipeline-parallelism-improve-stage1-1 ping -c 10 172.32.0.4 >> $OUTPUT_DIR/connectivity.log 2>&1
}

# Monitor bandwidth usage
monitor_bandwidth() {
    echo "Monitoring bandwidth usage..."
    
    # Start iftop monitoring in each container
    docker exec -d pipeline-parallelism-improve-stage1-1 sh -c "
        while true; do
            cat /proc/net/dev | grep eth0 >> /tmp/network_stats.log
            sleep 1
        done
    " &
    
    docker exec -d pipeline-parallelism-improve-stage2-1 sh -c "
        while true; do
            cat /proc/net/dev | grep eth0 >> /tmp/network_stats.log
            sleep 1
        done
    " &
    
    docker exec -d pipeline-parallelism-improve-coordinator-1 sh -c "
        while true; do
            cat /proc/net/dev | grep eth0 >> /tmp/network_stats.log
            sleep 1
        done
    " &
}

# Monitor communication patterns
monitor_communication() {
    echo "Monitoring communication patterns..."
    
    # Use tcpdump to capture traffic between containers
    docker exec -d pipeline-parallelism-improve-stage1-1 tcpdump -i eth0 -w /tmp/stage1_traffic.pcap host 172.32.0.3 or host 172.32.0.4 2>/dev/null &
    docker exec -d pipeline-parallelism-improve-stage2-1 tcpdump -i eth0 -w /tmp/stage2_traffic.pcap host 172.32.0.2 or host 172.32.0.4 2>/dev/null &
    docker exec -d pipeline-parallelism-improve-coordinator-1 tcpdump -i eth0 -w /tmp/coordinator_traffic.pcap host 172.32.0.2 or host 172.32.0.3 2>/dev/null &
}

# Performance benchmark
performance_benchmark() {
    echo "Running performance benchmark..."
    
    # Test throughput between stages using iperf3 if available
    echo "Testing throughput stage1 -> stage2..."
    docker exec -d pipeline-parallelism-improve-stage2-1 iperf3 -s -p 5001 2>/dev/null &
    sleep 2
    docker exec pipeline-parallelism-improve-stage1-1 iperf3 -c 172.32.0.3 -p 5001 -t 10 > $OUTPUT_DIR/throughput_stage1_stage2.log 2>&1
    
    echo "Testing throughput stage2 -> coordinator..."
    docker exec -d pipeline-parallelism-improve-coordinator-1 iperf3 -s -p 5002 2>/dev/null &
    sleep 2
    docker exec pipeline-parallelism-improve-stage2-1 iperf3 -c 172.32.0.4 -p 5002 -t 10 > $OUTPUT_DIR/throughput_stage2_coordinator.log 2>&1
}

# Start monitoring
echo "$(date): Starting network monitoring..." > $OUTPUT_DIR/monitoring.log

test_connectivity
monitor_bandwidth
monitor_communication

# Run for specified duration
echo "Monitoring for $DURATION seconds..."
sleep $DURATION

# Collect results
echo "Collecting monitoring results..."

# Copy network stats from containers
docker cp pipeline-parallelism-improve-stage1-1:/tmp/network_stats.log $OUTPUT_DIR/stage1_network_stats.log 2>/dev/null
docker cp pipeline-parallelism-improve-stage2-1:/tmp/network_stats.log $OUTPUT_DIR/stage2_network_stats.log 2>/dev/null
docker cp pipeline-parallelism-improve-coordinator-1:/tmp/network_stats.log $OUTPUT_DIR/coordinator_network_stats.log 2>/dev/null

# Copy traffic captures
docker cp pipeline-parallelism-improve-stage1-1:/tmp/stage1_traffic.pcap $OUTPUT_DIR/ 2>/dev/null
docker cp pipeline-parallelism-improve-stage2-1:/tmp/stage2_traffic.pcap $OUTPUT_DIR/ 2>/dev/null
docker cp pipeline-parallelism-improve-coordinator-1:/tmp/coordinator_traffic.pcap $OUTPUT_DIR/ 2>/dev/null

# Show current network configuration
echo "=== Current Network Configuration ===" >> $OUTPUT_DIR/network_config.log
docker exec pipeline-parallelism-improve-stage1-1 tc qdisc show dev eth0 >> $OUTPUT_DIR/network_config.log 2>&1
docker exec pipeline-parallelism-improve-stage2-1 tc qdisc show dev eth0 >> $OUTPUT_DIR/network_config.log 2>&1
docker exec pipeline-parallelism-improve-coordinator-1 tc qdisc show dev eth0 >> $OUTPUT_DIR/network_config.log 2>&1

echo "$(date): Monitoring completed. Results saved in $OUTPUT_DIR/" >> $OUTPUT_DIR/monitoring.log
echo "Monitoring completed! Results saved in $OUTPUT_DIR/"

# Generate summary report
echo "=== Network Monitoring Summary ===" > $OUTPUT_DIR/summary.txt
echo "Duration: $DURATION seconds" >> $OUTPUT_DIR/summary.txt
echo "Generated: $(date)" >> $OUTPUT_DIR/summary.txt
echo "" >> $OUTPUT_DIR/summary.txt
echo "Files generated:" >> $OUTPUT_DIR/summary.txt
ls -la $OUTPUT_DIR/ >> $OUTPUT_DIR/summary.txt 