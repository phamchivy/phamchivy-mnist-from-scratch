#!/bin/bash

# debug_hybrid.sh - Debug Script for Hybrid System Issues
# Usage: ./debug_hybrid.sh

set -e

echo "🔍 HYBRID SYSTEM DEBUG DIAGNOSTICS"
echo "=================================="

# Function to check container status
check_container_status() {
    echo "📊 Container Status:"
    docker-compose -f docker-compose-hybrid.yml ps
    echo ""
}

# Function to check network connectivity
check_network_connectivity() {
    echo "🌐 Network Connectivity Test:"
    
    echo "Testing Stage1-1 → Stage2-1:"
    docker exec distributed_dnn-stage1-1-1 nc -zv 172.33.0.4 8001 || echo "❌ Failed"
    
    echo "Testing Stage1-2 → Stage2-1:"
    docker exec distributed_dnn-stage1-2-1 nc -zv 172.33.0.4 8002 || echo "❌ Failed"
    
    echo "Testing Stage2-1 → Stage1-1 backward:"
    docker exec distributed_dnn-stage2-1-1 nc -zv 172.33.0.2 8003 || echo "❌ Failed"
    
    echo "Testing Stage2-1 → Stage1-2 backward:"
    docker exec distributed_dnn-stage2-1-1 nc -zv 172.33.0.3 8004 || echo "❌ Failed"
    
    echo "Testing Workers → Coordinator EASGD:"
    docker exec distributed_dnn-stage1-1-1 nc -zv 172.33.0.5 12348 || echo "❌ Failed"
    docker exec distributed_dnn-stage1-2-1 nc -zv 172.33.0.5 12348 || echo "❌ Failed"
    docker exec distributed_dnn-stage2-1-1 nc -zv 172.33.0.5 12348 || echo "❌ Failed"
    
    echo ""
}

# Function to check data loading
check_data_loading() {
    echo "📁 Data Loading Check:"
    
    echo "Checking if MNIST data exists in containers:"
    docker exec distributed_dnn-stage1-1-1 ls -la /app/data/ || echo "❌ No data in Stage1-1"
    docker exec distributed_dnn-stage1-2-1 ls -la /app/data/ || echo "❌ No data in Stage1-2"
    
    echo "Checking file sizes:"
    docker exec distributed_dnn-stage1-1-1 wc -l /app/data/mnist_train.csv || echo "❌ Cannot count lines"
    
    echo ""
}

# Function to check process status
check_processes() {
    echo "🔄 Process Status:"
    
    echo "Stage1-1 processes:"
    docker exec distributed_dnn-stage1-1-1 ps aux | grep app_hybrid || echo "❌ No hybrid process"
    
    echo "Stage1-2 processes:"
    docker exec distributed_dnn-stage1-2-1 ps aux | grep app_hybrid || echo "❌ No hybrid process"
    
    echo "Stage2-1 processes:"
    docker exec distributed_dnn-stage2-1-1 ps aux | grep app_hybrid || echo "❌ No hybrid process"
    
    echo "Coordinator processes:"
    docker exec distributed_dnn-coordinator-1 ps aux | grep app_hybrid || echo "❌ No hybrid process"
    
    echo ""
}

# Function to check socket status
check_sockets() {
    echo "🔌 Socket Status:"
    
    echo "Stage1-1 sockets:"
    docker exec distributed_dnn-stage1-1-1 netstat -tlnp | grep ":800[3-4]" || echo "❌ No backward servers"
    
    echo "Stage1-2 sockets:"
    docker exec distributed_dnn-stage1-2-1 netstat -tlnp | grep ":800[3-4]" || echo "❌ No backward servers"
    
    echo "Stage2-1 sockets:"
    docker exec distributed_dnn-stage2-1-1 netstat -tlnp | grep ":800[1-2]" || echo "❌ No forward servers"
    
    echo "Coordinator sockets:"
    docker exec distributed_dnn-coordinator-1 netstat -tlnp | grep ":1234[7-8]" || echo "❌ No EASGD/stats servers"
    
    echo ""
}

# Function to extract and analyze logs
analyze_logs() {
    echo "📋 Log Analysis:"
    
    echo "=== Recent Stage1-1 Errors ==="
    docker logs distributed_dnn-stage1-1-1 2>&1 | tail -20 | grep -E "(Error|Failed|Invalid)" || echo "No errors found"
    
    echo "=== Recent Stage1-2 Errors ==="
    docker logs distributed_dnn-stage1-2-1 2>&1 | tail -20 | grep -E "(Error|Failed|Invalid)" || echo "No errors found"
    
    echo "=== Recent Stage2-1 Errors ==="
    docker logs distributed_dnn-stage2-1-1 2>&1 | tail -20 | grep -E "(Error|Failed|Invalid)" || echo "No errors found"
    
    echo "=== Recent Coordinator Errors ==="
    docker logs distributed_dnn-coordinator-1 2>&1 | tail -20 | grep -E "(Error|Failed|Invalid)" || echo "No errors found"
    
    echo ""
}

# Function to check memory usage
check_memory() {
    echo "💾 Memory Usage:"
    
    echo "Stage1-1 memory:"
    docker exec distributed_dnn-stage1-1-1 free -h
    
    echo "Stage1-2 memory:"
    docker exec distributed_dnn-stage1-2-1 free -h
    
    echo "Stage2-1 memory:"
    docker exec distributed_dnn-stage2-1-1 free -h
    
    echo "Coordinator memory:"
    docker exec distributed_dnn-coordinator-1 free -h
    
    echo ""
}

# Function to test EASGD message flow
test_easgd_communication() {
    echo "📡 EASGD Communication Test:"
    
    echo "Testing if coordinator EASGD port is accessible:"
    timeout 5 docker exec distributed_dnn-stage1-1-1 nc -zv 172.33.0.5 12348 && echo "✅ EASGD port accessible" || echo "❌ EASGD port not accessible"
    
    echo "Checking coordinator EASGD server status:"
    docker exec distributed_dnn-coordinator-1 netstat -tlnp | grep ":12348" && echo "✅ EASGD server listening" || echo "❌ EASGD server not listening"
    
    echo ""
}

# Function to extract training progress
check_training_progress() {
    echo "📈 Training Progress Check:"
    
    echo "=== Stage1-1 Training Progress ==="
    docker logs distributed_dnn-stage1-1-1 2>&1 | grep -E "(Epoch|batches|Flattened)" | tail -10
    
    echo "=== Stage1-2 Training Progress ==="
    docker logs distributed_dnn-stage1-2-1 2>&1 | grep -E "(Epoch|batches|Flattened)" | tail -10
    
    echo "=== Stage2-1 Training Progress ==="
    docker logs distributed_dnn-stage2-1-1 2>&1 | grep -E "(Epoch|Processed|messages)" | tail -10
    
    echo "=== Coordinator Progress ==="
    docker logs distributed_dnn-coordinator-1 2>&1 | grep -E "(Epoch|coordination|divergence)" | tail -10
    
    echo ""
}

# Function to check threading issues
check_threading() {
    echo "🧵 Threading Status:"
    
    echo "Stage1-1 thread count:"
    docker exec distributed_dnn-stage1-1-1 ps -eLf | grep app_hybrid | wc -l || echo "❌ Cannot count threads"
    
    echo "Stage2-1 thread count:"
    docker exec distributed_dnn-stage2-1-1 ps -eLf | grep app_hybrid | wc -l || echo "❌ Cannot count threads"
    
    echo ""
}

# Function to diagnose specific issues
diagnose_issues() {
    echo "🩺 Issue Diagnosis:"
    
    # Check if containers are stuck in a loop
    echo "Checking for infinite loops (CPU usage):"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" distributed_dnn-stage1-1-1 distributed_dnn-stage1-2-1 distributed_dnn-stage2-1-1 distributed_dnn-coordinator-1
    
    # Check file descriptor usage
    echo "Checking file descriptor usage:"
    docker exec distributed_dnn-coordinator-1 ls /proc/self/fd | wc -l || echo "Cannot check FDs"
    
    # Check for socket leaks
    echo "Checking for socket leaks:"
    docker exec distributed_dnn-coordinator-1 netstat -an | grep -c "ESTABLISHED\|LISTEN" || echo "Cannot check sockets"
    
    echo ""
}

# Function to provide fix recommendations
recommend_fixes() {
    echo "🔧 RECOMMENDED FIXES:"
    echo "==================="
    
    echo "1. Pipeline Training Issues:"
    echo "   - Check if data is properly loaded and partitioned"
    echo "   - Verify forward/backward thread synchronization"
    echo "   - Ensure Stage1 workers are actually processing data"
    
    echo ""
    echo "2. EASGD Socket Issues:"
    echo "   - Implement per-epoch socket creation/cleanup"
    echo "   - Add better timeout handling for socket operations"
    echo "   - Improve error recovery for failed connections"
    
    echo ""
    echo "3. Threading Problems:"
    echo "   - Verify global_training_active flag management"
    echo "   - Check thread lifecycle and cleanup"
    echo "   - Ensure proper condition variable signaling"
    
    echo ""
    echo "4. Immediate Actions:"
    echo "   - Restart the system: docker-compose -f docker-compose-hybrid.yml restart"
    echo "   - Clear containers: docker-compose -f docker-compose-hybrid.yml down && docker-compose -f docker-compose-hybrid.yml up"
    echo "   - Check disk space: df -h"
    echo "   - Monitor real-time: docker-compose -f docker-compose-hybrid.yml logs -f"
    
    echo ""
}

# Function to generate debug report
generate_debug_report() {
    echo "📝 Generating Debug Report..."
    
    REPORT_FILE="hybrid_debug_report_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "HYBRID SYSTEM DEBUG REPORT"
        echo "Generated: $(date)"
        echo "=========================="
        echo ""
        
        echo "=== CONTAINER STATUS ==="
        docker-compose -f docker-compose-hybrid.yml ps
        echo ""
        
        echo "=== RECENT LOGS (Stage1-1) ==="
        docker logs distributed_dnn-stage1-1-1 2>&1 | tail -50
        echo ""
        
        echo "=== RECENT LOGS (Stage1-2) ==="
        docker logs distributed_dnn-stage1-2-1 2>&1 | tail -50
        echo ""
        
        echo "=== RECENT LOGS (Stage2-1) ==="
        docker logs distributed_dnn-stage2-1-1 2>&1 | tail -50
        echo ""
        
        echo "=== RECENT LOGS (Coordinator) ==="
        docker logs distributed_dnn-coordinator-1 2>&1 | tail -50
        echo ""
        
        echo "=== NETWORK STATUS ==="
        docker network inspect hybrid_pipeline_network
        echo ""
        
        echo "=== SYSTEM RESOURCES ==="
        docker stats --no-stream
        echo ""
        
    } > "$REPORT_FILE"
    
    echo "Debug report saved to: $REPORT_FILE"
    echo ""
}

# Function to run quick fixes
run_quick_fixes() {
    echo "⚡ Running Quick Fixes..."
    
    # 1. Restart containers in correct order
    echo "1. Restarting coordinator first..."
    docker-compose -f docker-compose-hybrid.yml restart coordinator
    sleep 5
    
    echo "2. Restarting Stage2..."
    docker-compose -f docker-compose-hybrid.yml restart stage2-1
    sleep 5
    
    echo "3. Restarting Stage1 workers..."
    docker-compose -f docker-compose-hybrid.yml restart stage1-1 stage1-2
    sleep 10
    
    # 2. Check if they come up properly
    echo "4. Checking container health..."
    sleep 15
    docker-compose -f docker-compose-hybrid.yml ps
    
    echo "Quick fixes completed. Monitor logs with:"
    echo "docker-compose -f docker-compose-hybrid.yml logs -f"
    echo ""
}

# Function to run comprehensive test
run_comprehensive_test() {
    echo "🧪 COMPREHENSIVE HYBRID SYSTEM TEST"
    echo "==================================="
    
    check_container_status
    check_data_loading
    check_processes
    check_sockets
    check_network_connectivity
    test_easgd_communication
    check_training_progress
    check_threading
    check_memory
    analyze_logs
    diagnose_issues
    recommend_fixes
}

# Main menu
show_menu() {
    echo ""
    echo "🔍 HYBRID SYSTEM DEBUG MENU"
    echo "==========================="
    echo "1. Run comprehensive test"
    echo "2. Check container status"
    echo "3. Check network connectivity"
    echo "4. Analyze logs"
    echo "5. Check training progress"
    echo "6. Test EASGD communication"
    echo "7. Generate debug report"
    echo "8. Run quick fixes"
    echo "9. Monitor live logs"
    echo "0. Exit"
    echo ""
    read -p "Select option (0-9): " choice
}

# Handle menu selection
handle_selection() {
    case $choice in
        1) run_comprehensive_test ;;
        2) check_container_status ;;
        3) check_network_connectivity ;;
        4) analyze_logs ;;
        5) check_training_progress ;;
        6) test_easgd_communication ;;
        7) generate_debug_report ;;
        8) run_quick_fixes ;;
        9) echo "Monitoring live logs (Ctrl+C to exit):"; docker-compose -f docker-compose-hybrid.yml logs -f ;;
        0) echo "Exiting debug script."; exit 0 ;;
        *) echo "Invalid option. Please try again." ;;
    esac
}

# Main script logic
main() {
    echo "🚀 Starting Hybrid System Debug Session..."
    echo "Checking if hybrid system is running..."
    
    if ! docker-compose -f docker-compose-hybrid.yml ps | grep -q "Up"; then
        echo "❌ Hybrid system is not running!"
        echo "Start it with: ./launch_hybrid.sh docker local"
        exit 1
    fi
    
    echo "✅ Hybrid system is running. Starting diagnostics..."
    echo ""
    
    while true; do
        show_menu
        handle_selection
        echo ""
        read -p "Press Enter to continue..."
    done
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi