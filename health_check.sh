#!/bin/bash
# Health check for hybrid system components

if pgrep -f "app_hybrid" > /dev/null; then
    # Check if sockets are listening (for coordinator and stage2)
    if [[ "$1" == "coordinator" ]]; then
        # Check if EASGD and stats ports are listening
        if netstat -tln | grep -q ":12348" && netstat -tln | grep -q ":12347"; then
            echo "Coordinator healthy: Both EASGD and stats servers running"
            exit 0
        else
            echo "Coordinator unhealthy: Required ports not listening"
            exit 1
        fi
    elif [[ "$1" == "stage2" ]]; then
        # Check if Stage2 ports are listening
        if netstat -tln | grep -q ":8001\|:8002"; then
            echo "Stage2 healthy: Pipeline servers running"
            exit 0
        else
            echo "Stage2 unhealthy: Pipeline ports not listening"
            exit 1
        fi
    else
        # For Stage1 workers, just check if process is running
        echo "Stage1 healthy: Process running"
        exit 0
    fi
else
    echo "Unhealthy: app_hybrid process not found"
    exit 1
fi