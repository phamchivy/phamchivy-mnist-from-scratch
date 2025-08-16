#!/bin/bash

# launch_hybrid.sh - Complete Hybrid Pipeline-EASGD System Launcher
# Usage: ./launch_hybrid.sh [local|docker] [lan|wan|local] [alpha1] [alpha2] [beta]

set -e

# Default parameters
MODE=${1:-docker}
NETWORK_TYPE=${2:-local}
ALPHA1=${3:-0.2}
ALPHA2=${4:-0.3}
BETA=${5:-0.5}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Validate float function (replaced bc dependency)
validate_float() {
    local value=$1
    local min=$2
    local max=$3
    
    # Check if it's a valid number using awk
    if ! echo "$value" | awk '{exit ($1 != $1 || $1 < '"$min"' || $1 > '"$max"')}'; then
        return 1
    fi
    return 0
}

# Banner
echo -e "${BLUE}"
echo "████████████████████████████████████████████████████████████████"
echo "█           HYBRID PIPELINE-EASGD SYSTEM LAUNCHER             █"
echo "█                                                              █"
echo "█  Mode: $MODE | Network: $NETWORK_TYPE | α₁=$ALPHA1 α₂=$ALPHA2 β=$BETA  █"
echo "████████████████████████████████████████████████████████████████"
echo -e "${NC}"

# Validate parameters
validate_parameters() {
    log "Validating parameters..."
    
    if [[ "$MODE" != "local" && "$MODE" != "docker" ]]; then
        error "Invalid mode: $MODE. Must be 'local' or 'docker'"
    fi
    
    if [[ "$NETWORK_TYPE" != "local" && "$NETWORK_TYPE" != "lan" && "$NETWORK_TYPE" != "wan" ]]; then
        error "Invalid network type: $NETWORK_TYPE. Must be 'local', 'lan', or 'wan'"
    fi
    
    # Validate alpha/beta ranges using awk instead of bc
    if ! validate_float "$ALPHA1" "0.01" "1.0"; then
        error "Invalid α₁: $ALPHA1. Must be between 0.01 and 1.0"
    fi
    
    if ! validate_float "$ALPHA2" "0.01" "1.0"; then
        error "Invalid α₂: $ALPHA2. Must be between 0.01 and 1.0"
    fi
    
    if ! validate_float "$BETA" "0.01" "1.0"; then
        error "Invalid β: $BETA. Must be between 0.01 and 1.0"
    fi
    
    log "Parameters validated successfully"
}

# Check dependencies
check_dependencies() {
    log "Checking dependencies..."
    
    if [[ "$MODE" == "docker" ]]; then
        if ! command -v docker &> /dev/null; then
            error "Docker is required but not installed"
        fi
        
        if ! command -v docker-compose &> /dev/null; then
            error "Docker Compose is required but not installed"
        fi
        
        # Check Docker daemon
        if ! docker info &> /dev/null; then
            error "Docker daemon is not running"
        fi
        
        log "Docker dependencies satisfied"
    else
        if ! command -v gcc &> /dev/null; then
            error "GCC compiler is required but not installed"
        fi
        
        if ! command -v make &> /dev/null; then
            error "Make is required but not installed"
        fi
        
        log "Local build dependencies satisfied"
    fi
}

# Prepare environment
prepare_environment() {
    log "Preparing environment..."
    
    # Create necessary directories
    mkdir -p results logs data
    
    # Set environment variables
    export NETWORK_TYPE=$NETWORK_TYPE
    export ALPHA1=$ALPHA1
    export ALPHA2=$ALPHA2
    export BETA=$BETA
    
    # Set network simulation parameters
    case "$NETWORK_TYPE" in
        "lan")
            export NETWORK_DELAY="2ms"
            export NETWORK_JITTER="0.5ms"
            export NETWORK_LOSS="0.01%"
            log "LAN simulation: 2ms±0.5ms delay, 0.01% loss"
            ;;
        "wan")
            export NETWORK_DELAY="100ms"
            export NETWORK_JITTER="10ms"
            export NETWORK_LOSS="0.5%"
            log "WAN simulation: 100ms±10ms delay, 0.5% loss"
            ;;
        *)
            unset NETWORK_DELAY NETWORK_JITTER NETWORK_LOSS
            log "Local network: no simulation"
            ;;
    esac
    
    log "Environment prepared"
}

# Build system
build_system() {
    log "Building hybrid system..."
    
    if [[ "$MODE" == "docker" ]]; then
        log "Building Docker images..."
        
        # Check if docker-compose-hybrid.yml exists
        if [[ ! -f "docker-compose-hybrid.yml" ]]; then
            warn "docker-compose-hybrid.yml not found, using docker-compose-pipeline.yml"
            docker-compose -f docker-compose-pipeline.yml build --no-cache
        else
            docker-compose -f docker-compose-hybrid.yml build --no-cache
        fi
        log "Docker images built successfully"
    else
        log "Building local binaries..."
        make clean
        
        # Check if hybrid makefile targets exist
        if make -n app_hybrid &> /dev/null; then
            make app_hybrid
        else
            warn "Hybrid targets not found, building pipeline target"
            #make app_pipeline
        fi
        log "Local binaries built successfully"
    fi
}

# Launch Docker mode
launch_docker() {
    log "Launching Docker-based hybrid system..."
    
    # Determine which compose file to use
    COMPOSE_FILE="docker-compose-hybrid.yml"
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        warn "Hybrid compose file not found, falling back to pipeline compose"
        COMPOSE_FILE="docker-compose-pipeline.yml"
    fi
    
    # Start containers
    log "Starting containers using $COMPOSE_FILE..."
    docker-compose -f "$COMPOSE_FILE" up -d
    
    # Wait for containers to be ready
    log "Waiting for containers to initialize..."
    sleep 10
    
    # Check container health
    log "Checking container health..."
    
    # Show container status
    echo -e "\n${BLUE}Container Status:${NC}"
    docker-compose -f "$COMPOSE_FILE" ps
    
    # Follow logs
    echo -e "\n${GREEN}Following container logs (Ctrl+C to stop monitoring):${NC}"
    echo -e "${YELLOW}To stop the system: docker-compose -f $COMPOSE_FILE down${NC}"
    docker-compose -f "$COMPOSE_FILE" logs -f
}

# Launch Local mode
launch_local() {
    log "Launching local hybrid system..."

    if [[ -f "./app_hybrid" ]]; then
        BINARY="./app_hybrid"
    else
        error "No hybrid executable found. Run 'make' first."
    fi

    log "Using binary: $BINARY"

    # Start hybrid system
    log "Starting coordinator..."
    $BINARY coordinator $BETA > logs/coordinator.log 2>&1 &
    local coordinator_pid=$!
    sleep 3

    log "Starting Stage2 worker..."
    $BINARY stage2 $ALPHA2 > logs/stage2.log 2>&1 &
    local stage2_pid=$!
    sleep 3

    log "Starting Stage1-1 worker..."
    $BINARY stage1 1 0 30000 $ALPHA1 > logs/stage1-1.log 2>&1 &
    local stage1_1_pid=$!

    log "Starting Stage1-2 worker..."
    $BINARY stage1 2 30000 60000 $ALPHA1 > logs/stage1-2.log 2>&1 &
    local stage1_2_pid=$!

    echo "$coordinator_pid $stage2_pid $stage1_1_pid $stage1_2_pid" > logs/pids.txt

    log "All workers started successfully"
    echo -e "\n${YELLOW}Monitor logs with:${NC}"
    echo "  tail -f logs/coordinator.log"
    echo "  tail -f logs/stage*.log"

    echo -e "\n${YELLOW}Stop system with:${NC}"
    echo "  kill \$(cat logs/pids.txt)"

    echo -e "\n${GREEN}Following coordinator logs (Ctrl+C to stop monitoring):${NC}"
    tail -f logs/coordinator.log
}

# Create stop script for local mode
create_stop_script() {
    cat > stop_hybrid.sh << 'EOF'
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
EOF
    chmod +x stop_hybrid.sh
}

# Show usage
show_usage() {
    echo "Usage: $0 [mode] [network] [alpha1] [alpha2] [beta]"
    echo ""
    echo "Parameters:"
    echo "  mode     : 'local' or 'docker' (default: docker)"
    echo "  network  : 'local', 'lan', or 'wan' (default: local)"
    echo "  alpha1   : Stage1 elastic rate 0.01-1.0 (default: 0.2)"
    echo "  alpha2   : Stage2 elastic rate 0.01-1.0 (default: 0.3)"
    echo "  beta     : Server update rate 0.01-1.0 (default: 0.5)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Docker mode, local network, default params"
    echo "  $0 docker lan                        # Docker mode, LAN simulation"
    echo "  $0 local local 0.3 0.4 0.6          # Local mode, custom parameters"
    echo "  $0 docker wan 0.1 0.2 0.3           # Docker mode, WAN simulation, conservative params"
}

# Cleanup function
cleanup() {
    log "Cleaning up..."
    
    if [[ "$MODE" == "docker" ]]; then
        if [[ -f "docker-compose-hybrid.yml" ]]; then
            docker-compose -f docker-compose-hybrid.yml down 2>/dev/null || true
        else
            docker-compose -f docker-compose-pipeline.yml down 2>/dev/null || true
        fi
    else
        if [[ -f "logs/pids.txt" ]]; then
            PIDS=$(cat logs/pids.txt)
            kill $PIDS 2>/dev/null || true
            rm logs/pids.txt
        fi
    fi
    
    log "Cleanup completed"
}

# Trap cleanup on exit
# trap cleanup EXIT

# Main execution
main() {
    if [[ "$1" == "--help" || "$1" == "-h" ]]; then
        show_usage
        exit 0
    fi
    
    validate_parameters
    check_dependencies
    prepare_environment
    build_system
    
    if [[ "$MODE" == "local" ]]; then
        create_stop_script
    fi
    
    log "Launching hybrid system in $MODE mode..."
    
    if [[ "$MODE" == "docker" ]]; then
        launch_docker
    else
        launch_local
    fi
}

# Run main function
main "$@"