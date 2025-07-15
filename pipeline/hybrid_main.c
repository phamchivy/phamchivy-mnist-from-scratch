/**
 * @file hybrid_main.c
 * @brief Main Entry Point for Hybrid Pipeline-EASGD System
 * @author Research Team
 * @date 2024
 * 
 * This file provides the main entry point for the hybrid system that combines
 * pipeline parallelism with EASGD. It supports multiple components:
 * - Stage1 workers (data partitioning + pipeline)
 * - Stage2 worker (multiplexed processing)
 * - Coordinator (parameter server + statistics)
 */

#include "hybrid_pipeline_easgd.h"
#include <signal.h>
#include <getopt.h>
#include <stdlib.h>      // malloc, free, getenv
#include <stdio.h>       // printf, fprintf, fopen, fclose, FILE, stdout, fflush
#include <string.h>      // memset, strcpy, strcmp
#include <netinet/in.h>  // struct sockaddr_in, socklen_t
#include <sys/socket.h>  // accept

/* =============================================================================
 * SIGNAL HANDLING
 * ============================================================================= */

static volatile int shutdown_requested = 0;

void signal_handler(int sig) {
    printf("\n[MAIN] Received signal %d, initiating graceful shutdown...\n", sig);
    shutdown_requested = 1;
}

void setup_signal_handlers(void) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGPIPE, SIG_IGN); // Ignore broken pipe signals
}

/* =============================================================================
 * USAGE AND HELP FUNCTIONS
 * ============================================================================= */

void print_usage(const char* program_name) {
    printf("Usage: %s <component> [options]\n\n", program_name);
    printf("Components:\n");
    printf("  stage1 <worker_id> <data_start> <data_end> <alpha1>\n");
    printf("    worker_id    : Worker identifier (1 or 2)\n");
    printf("    data_start   : Start index in dataset (0 or 30000)\n");
    printf("    data_end     : End index in dataset (30000 or 60000)\n");
    printf("    alpha1       : Elastic averaging rate for Stage1 (0.1-0.5)\n");
    printf("\n");
    printf("  stage2 <alpha2>\n");
    printf("    alpha2       : Elastic averaging rate for Stage2 (0.1-0.5)\n");
    printf("\n");
    printf("  coordinator <beta>\n");
    printf("    beta         : Server update rate for EASGD (0.1-0.9)\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s stage1 1 0 30000 0.2\n", program_name);
    printf("  %s stage1 2 30000 60000 0.2\n", program_name);
    printf("  %s stage2 0.3\n", program_name);
    printf("  %s coordinator 0.5\n", program_name);
    printf("\n");
    printf("Environment Variables:\n");
    printf("  NETWORK_TYPE  : Network environment (LOCAL/LAN/WAN)\n");
    printf("  LOG_LEVEL     : Logging verbosity (0-3)\n");
    printf("\n");
}

void print_system_info(void) {
    printf("=== HYBRID PIPELINE-EASGD SYSTEM ===\n");
    printf("Version: 2.0\n");
    printf("Architecture: Pipeline Parallelism + EASGD\n");
    printf("Dataset: MNIST (60,000 training + 10,000 test)\n");
    printf("Network: 784 → 512 → 10 (2-stage pipeline)\n");
    
    const char* network_type = getenv("NETWORK_TYPE");
    if (network_type) {
        printf("Network Environment: %s\n", network_type);
    } else {
        printf("Network Environment: LOCAL (default)\n");
    }
    
    printf("=====================================\n\n");
}

/* =============================================================================
 * PARAMETER VALIDATION
 * ============================================================================= */

int validate_stage1_parameters(int worker_id, int data_start, int data_end, double alpha1) {
    if (worker_id < 1 || worker_id > 2) {
        printf("[ERROR] Invalid worker_id: %d (must be 1 or 2)\n", worker_id);
        return -1;
    }
    
    if (data_start < 0 || data_end <= data_start || data_end > 60000) {
        printf("[ERROR] Invalid data range: [%d:%d] (must be within [0:60000])\n", 
               data_start, data_end);
        return -1;
    }
    
    // Validate expected data partitions
    if ((worker_id == 1 && (data_start != 0 || data_end != 30000)) ||
        (worker_id == 2 && (data_start != 30000 || data_end != 60000))) {
        printf("[WARNING] Unexpected data partition for worker %d: [%d:%d]\n", 
               worker_id, data_start, data_end);
        printf("[WARNING] Expected: Worker 1=[0:30000], Worker 2=[30000:60000]\n");
    }
    
    if (alpha1 < 0.01 || alpha1 > 1.0) {
        printf("[ERROR] Invalid alpha1: %.3f (must be between 0.01 and 1.0)\n", alpha1);
        return -1;
    }
    
    return 0;
}

int validate_stage2_parameters(double alpha2) {
    if (alpha2 < 0.01 || alpha2 > 1.0) {
        printf("[ERROR] Invalid alpha2: %.3f (must be between 0.01 and 1.0)\n", alpha2);
        return -1;
    }
    
    return 0;
}

int validate_coordinator_parameters(double beta) {
    if (beta < 0.01 || beta > 1.0) {
        printf("[ERROR] Invalid beta: %.3f (must be between 0.01 and 1.0)\n", beta);
        return -1;
    }
    
    return 0;
}

/* =============================================================================
 * COMPONENT LAUNCHER FUNCTIONS
 * ============================================================================= */

int launch_stage1_worker(int worker_id, int data_start, int data_end, double alpha1) {
    printf("[STAGE1-%d] Launching worker with parameters:\n", worker_id);
    printf("  Data partition: [%d:%d] (%d samples)\n", data_start, data_end, data_end - data_start);
    printf("  Alpha1 (elastic rate): %.3f\n", alpha1);
    printf("  Expected connections: Stage2-1, Coordinator\n\n");
    
    // Set optimal parameters based on network type
    const char* network_type = getenv("NETWORK_TYPE");
    if (network_type) {
        double optimal_alpha1, optimal_alpha2, optimal_beta;
        set_optimal_hybrid_parameters(&optimal_alpha1, &optimal_alpha2, &optimal_beta, network_type);
        
        if (fabs(alpha1 - optimal_alpha1) > 0.05) {
            printf("[WARNING] Non-optimal alpha1 for %s network (recommended: %.3f)\n", 
                   network_type, optimal_alpha1);
        }
    }
    
    // Launch Stage1 worker
    hybrid_stage1_main(worker_id, data_start, data_end, alpha1);
    
    return 0;
}

int launch_stage2_worker(double alpha2) {
    printf("[STAGE2-1] Launching worker with parameters:\n");
    printf("  Alpha2 (elastic rate): %.3f\n", alpha2);
    printf("  Expected connections: Stage1-1, Stage1-2, Coordinator\n");
    printf("  Processing mode: Multiplexed pipeline\n\n");
    
    // Set optimal parameters based on network type
    const char* network_type = getenv("NETWORK_TYPE");
    if (network_type) {
        double optimal_alpha1, optimal_alpha2, optimal_beta;
        set_optimal_hybrid_parameters(&optimal_alpha1, &optimal_alpha2, &optimal_beta, network_type);
        
        if (fabs(alpha2 - optimal_alpha2) > 0.05) {
            printf("[WARNING] Non-optimal alpha2 for %s network (recommended: %.3f)\n", 
                   network_type, optimal_alpha2);
        }
    }
    
    // Launch Stage2 worker
    hybrid_stage2_main(alpha2);
    
    return 0;
}

int launch_coordinator(double beta) {
    printf("[COORDINATOR] Launching coordinator with parameters:\n");
    printf("  Beta (server rate): %.3f\n", beta);
    printf("  Expected workers: 2 Stage1 + 1 Stage2 = 3 total\n");
    printf("  Services: EASGD Parameter Server + Statistics Collection\n");
    printf("  Ports: %d (EASGD), 12347 (Statistics)\n\n", EASGD_COMM_PORT);
    
    // Validate hybrid configuration
    if (validate_hybrid_configuration(2, 1) < 0) {
        printf("[ERROR] Invalid hybrid configuration\n");
        return -1;
    }
    
    // Set optimal parameters based on network type
    const char* network_type = getenv("NETWORK_TYPE");
    if (network_type) {
        double optimal_alpha1, optimal_alpha2, optimal_beta;
        set_optimal_hybrid_parameters(&optimal_alpha1, &optimal_alpha2, &optimal_beta, network_type);
        
        if (fabs(beta - optimal_beta) > 0.05) {
            printf("[WARNING] Non-optimal beta for %s network (recommended: %.3f)\n", 
                   network_type, optimal_beta);
        }
    }
    
    // Launch coordinator
    hybrid_coordinator_main(beta);
    
    return 0;
}

/* =============================================================================
 * MAIN FUNCTION
 * ============================================================================= */

int main(int argc, char* argv[]) {
    // Setup signal handlers for graceful shutdown
    setup_signal_handlers();
    
    // Print system information
    print_system_info();
    
    // Check minimum arguments
    if (argc < 2) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    
    const char* component = argv[1];
    
    // Handle help requests
    if (strcmp(component, "--help") == 0 || strcmp(component, "-h") == 0) {
        print_usage(argv[0]);
        return EXIT_SUCCESS;
    }
    
    printf("[MAIN] Launching component: %s\n", component);
    
    /* Stage1 Worker */
    if (strcmp(component, "stage1") == 0) {
        if (argc != 6) {
            printf("[ERROR] Invalid arguments for stage1 component\n");
            printf("Usage: %s stage1 <worker_id> <data_start> <data_end> <alpha1>\n", argv[0]);
            return EXIT_FAILURE;
        }
        
        int worker_id = atoi(argv[2]);
        int data_start = atoi(argv[3]);
        int data_end = atoi(argv[4]);
        double alpha1 = atof(argv[5]);
        
        if (validate_stage1_parameters(worker_id, data_start, data_end, alpha1) < 0) {
            return EXIT_FAILURE;
        }
        
        return launch_stage1_worker(worker_id, data_start, data_end, alpha1);
    }
    
    /* Stage2 Worker */
    else if (strcmp(component, "stage2") == 0) {
        if (argc != 3) {
            printf("[ERROR] Invalid arguments for stage2 component\n");
            printf("Usage: %s stage2 <alpha2>\n", argv[0]);
            return EXIT_FAILURE;
        }
        
        double alpha2 = atof(argv[2]);
        
        if (validate_stage2_parameters(alpha2) < 0) {
            return EXIT_FAILURE;
        }
        
        return launch_stage2_worker(alpha2);
    }
    
    /* Coordinator */
    else if (strcmp(component, "coordinator") == 0) {
        if (argc != 3) {
            printf("[ERROR] Invalid arguments for coordinator component\n");
            printf("Usage: %s coordinator <beta>\n", argv[0]);
            return EXIT_FAILURE;
        }
        
        double beta = atof(argv[2]);
        
        if (validate_coordinator_parameters(beta) < 0) {
            return EXIT_FAILURE;
        }
        
        return launch_coordinator(beta);
    }
    
    /* Invalid component */
    else {
        printf("[ERROR] Unknown component: %s\n", component);
        printf("Valid components: stage1, stage2, coordinator\n\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
}