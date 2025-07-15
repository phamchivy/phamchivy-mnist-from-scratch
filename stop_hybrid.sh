#!/bin/bash

echo "Stopping hybrid system..."

if [[ -f "logs/pids.txt" ]]; then
    PIDS=$(cat logs/pids.txt)
    echo "Killing processes: $PIDS"
    kill $PIDS 2>/dev/null || true
    
    # Wait for graceful shutdown
    sleep 5
    
    # Force kill if still running
    kill -9 $PIDS 2>/dev/null || true
    
    rm logs/pids.txt
    echo "System stopped"
else
    echo "No PID file found. Trying to kill by process name..."
    pkill -f "app_hybrid\|app_pipeline" || true
    echo "Done"
fi
