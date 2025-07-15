#!/bin/bash

echo "=== Hybrid Pipeline-EASGD Startup Script ==="
echo "Container: $HOSTNAME"
echo "Network Type: ${NETWORK_TYPE:-LOCAL}"
echo "Component: $1"
echo "Parameters: ${@:2}"
echo "=========================================="

# Apply network simulation if configured
if [ ! -z "$NETWORK_DELAY" ]; then
    echo "Applying network simulation: delay=$NETWORK_DELAY, jitter=$NETWORK_JITTER, loss=$NETWORK_LOSS"
    tc qdisc add dev eth0 root netem delay $NETWORK_DELAY $NETWORK_JITTER loss $NETWORK_LOSS 2>/dev/null || true
fi

# Set optimal parameters based on network type
case "${NETWORK_TYPE:-LOCAL}" in
    "LAN")
        export ALPHA1=${ALPHA1:-0.3}
        export ALPHA2=${ALPHA2:-0.4}
        export BETA=${BETA:-0.6}
        echo "Using LAN-optimized parameters: α₁=$ALPHA1, α₂=$ALPHA2, β=$BETA"
        ;;
    "WAN")
        export ALPHA1=${ALPHA1:-0.1}
        export ALPHA2=${ALPHA2:-0.2}
        export BETA=${BETA:-0.3}
        echo "Using WAN-optimized parameters: α₁=$ALPHA1, α₂=$ALPHA2, β=$BETA"
        ;;
    *)
        export ALPHA1=${ALPHA1:-0.2}
        export ALPHA2=${ALPHA2:-0.3}
        export BETA=${BETA:-0.5}
        echo "Using LOCAL-optimized parameters: α₁=$ALPHA1, α₂=$ALPHA2, β=$BETA"
        ;;
esac

# Start monitoring in background
if [ "${LOG_LEVEL:-1}" -ge "2" ]; then
    echo "Starting system monitoring..."
    (while true; do
        echo "$(date): CPU=$(cat /proc/loadavg | cut -d' ' -f1), MEM=$(free -m | grep Mem | awk '{print $3"/"$2}')MB" >> /app/logs/system_monitor.log
        sleep 30
    done) &
fi

# Execute the hybrid application
echo "Starting hybrid application: $*"
exec ./app_hybrid "$@"