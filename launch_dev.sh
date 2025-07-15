#!/bin/bash

# Development Launch Script for Hybrid Pipeline-EASGD System
# Uses volume mounting to avoid rebuilding on code changes

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default parameters
MODE=${1:-docker}
NETWORK_TYPE=${2:-local}
ALPHA1=${3:-0.2}
ALPHA2=${4:-0.3}
BETA=${5:-0.5}

# Log function
log() {
    echo -e "${GREEN}[DEV]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Validate parameters
validate_parameters() {
    log "Validating parameters..."
    
    if [[ "$MODE" != "docker" && "$MODE" != "local" ]]; then
        error "Invalid mode: $MODE (use 'docker' or 'local')"
        exit 1
    fi
    
    if [[ "$NETWORK_TYPE" != "local" && "$NETWORK_TYPE" != "lan" && "$NETWORK_TYPE" != "wan" ]]; then
        error "Invalid network type: $NETWORK_TYPE (use 'local', 'lan', or 'wan')"
        exit 1
    fi
    
    # Validate EASGD parameters
    if (( $(echo "$ALPHA1 < 0.01 || $ALPHA1 > 1.0" | bc -l) )); then
        error "Invalid ALPHA1: $ALPHA1 (must be between 0.01 and 1.0)"
        exit 1
    fi
    
    if (( $(echo "$ALPHA2 < 0.01 || $ALPHA2 > 1.0" | bc -l) )); then
        error "Invalid ALPHA2: $ALPHA2 (must be between 0.01 and 1.0)"
        exit 1
    fi
    
    if (( $(echo "$BETA < 0.01 || $BETA > 1.0" | bc -l) )); then
        error "Invalid BETA: $BETA (must be between 0.01 and 1.0)"
        exit 1
    fi
    
    log "Parameters validated successfully"
}

# Check dependencies
check_dependencies() {
    log "Checking dependencies..."
    
    if [[ "$MODE" == "docker" ]]; then
        if ! command -v docker &> /dev/null; then
            error "Docker is not installed"
            exit 1
        fi
        
        if ! command -v docker-compose &> /dev/null; then
            error "Docker Compose is not installed"
            exit 1
        fi
    else
        if ! command -v make &> /dev/null; then
            error "Make is not installed"
            exit 1
        fi
        
        if ! command -v gcc &> /dev/null; then
            error "GCC is not installed"
            exit 1
        fi
    fi
    
    log "Dependencies check passed"
}

# Prepare environment
prepare_environment() {
    log "Preparing development environment..."
    
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
    
    log "Development environment prepared"
}

# Build development system
build_dev_system() {
    log "Building development system..."
    
    if [[ "$MODE" == "docker" ]]; then
        log "Building development Docker images..."
        
        # Check if docker-compose-dev.yml exists
        if [[ ! -f "docker-compose-dev.yml" ]]; then
            error "docker-compose-dev.yml not found"
            exit 1
        fi
        
        # Build only once (no --no-cache for faster builds)
        docker-compose -f docker-compose-dev.yml build
        log "Development Docker images built successfully"
    else
        log "Building local binaries..."
        make clean
        make app_hybrid
        log "Local binaries built successfully"
    fi
}

# Launch Docker development mode
launch_docker_dev() {
    log "Launching Docker-based development system..."
    
    # Start containers
    log "Starting development containers..."
    docker-compose -f docker-compose-dev.yml up -d
    
    # Wait for containers to be ready
    log "Waiting for containers to initialize..."
    sleep 15
    
    # Check container health
    log "Checking container health..."
    
    # Show container status
    echo -e "\n${BLUE}Container Status:${NC}"
    docker-compose -f docker-compose-dev.yml ps
    
    # Follow logs
    echo -e "\n${GREEN}Following container logs (Ctrl+C to stop monitoring):${NC}"
    echo -e "${YELLOW}To stop the system: docker-compose -f docker-compose-dev.yml down${NC}"
    echo -e "${YELLOW}To rebuild after code changes: docker-compose -f docker-compose-dev.yml restart${NC}"
    docker-compose -f docker-compose-dev.yml logs -f
}

# Launch Local development mode
launch_local_dev() {
    log "Launching local development system..."

    if [[ -f "./app_hybrid" ]]; then
        BINARY="./app_hybrid"
    else
        error "No hybrid executable found. Run 'make app_hybrid' first."
        exit 1
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

# Show usage
show_usage() {
    echo "Usage: $0 [mode] [network] [alpha1] [alpha2] [beta]"
    echo ""
    echo "Development Mode - Fast iteration with volume mounting"
    echo ""
    echo "Parameters:"
    echo "  mode     : 'local' or 'docker' (default: docker)"
    echo "  network  : 'local', 'lan', or 'wan' (default: local)"
    echo "  alpha1   : Stage1 elastic rate 0.01-1.0 (default: 0.2)"
    echo "  alpha2   : Stage2 elastic rate 0.01-1.0 (default: 0.3)"
    echo "  beta     : Server update rate 0.01-1.0 (default: 0.5)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Docker dev mode, local network, default params"
    echo "  $0 docker lan                        # Docker dev mode, LAN simulation"
    echo "  $0 local local 0.3 0.4 0.6          # Local dev mode, custom parameters"
    echo ""
    echo "Development Workflow:"
    echo "  1. Edit code in your IDE"
    echo "  2. Run: $0 docker local              # First time setup"
    echo "  3. After code changes: docker-compose -f docker-compose-dev.yml restart"
    echo "  4. No need to rebuild Docker images!"
}

# Cleanup function
cleanup() {
    log "Cleaning up development environment..."
    
    if [[ "$MODE" == "docker" ]]; then
        docker-compose -f docker-compose-dev.yml down 2>/dev/null || true
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
trap cleanup EXIT

# Main execution
main() {
    if [[ "$1" == "--help" || "$1" == "-h" ]]; then
        show_usage
        exit 0
    fi
    
    validate_parameters
    check_dependencies
    prepare_environment
    build_dev_system
    
    log "Launching development system in $MODE mode..."
    
    if [[ "$MODE" == "docker" ]]; then
        launch_docker_dev
    else
        launch_local_dev
    fi
}

# Run main function
main "$@" 