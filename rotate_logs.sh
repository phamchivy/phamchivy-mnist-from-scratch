#!/bin/bash
# Simple log rotation for hybrid system

LOG_DIR="/app/logs"
MAX_SIZE="100M"

for logfile in "$LOG_DIR"/*.log; do
    if [ -f "$logfile" ] && [ $(stat -c%s "$logfile" 2>/dev/null || echo 0) -gt 104857600 ]; then
        echo "Rotating large log file: $logfile"
        mv "$logfile" "${logfile}.old"
        touch "$logfile"
    fi
done