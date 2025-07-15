/**
 * @file easgd_core.c
 * @brief Core EASGD (Elastic Averaging SGD) Algorithm Implementation
 * @author Research Team
 * @date 2024
 * 
 * This file implements the core EASGD algorithm for distributed training:
 * 1. Master weight updates: w̄ ← w̄ + β(w_worker - w̄)
 * 2. Worker elastic averaging: w_worker ← w_worker + α(w̄ - w_worker)
 * 3. Weight divergence calculation and monitoring
 */

#include "hybrid_pipeline_easgd.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>      // malloc, free
#include <stdio.h>       // printf, perror, fflush, stdout
#include <sys/types.h>   // ssize_t
#include <sys/socket.h>  // send, recv, MSG_WAITALL

/* =============================================================================
 * EASGD MESSAGE HANDLING
 * ============================================================================= */

EASGDMessage* create_easgd_message(EASGDMessageType type, int worker_id, int stage_type) {
    EASGDMessage* msg = (EASGDMessage*)malloc(sizeof(EASGDMessage));
    if (!msg) return NULL;
    
    msg->message_type = type;
    msg->worker_id = worker_id;
    msg->stage_type = stage_type;
    msg->epoch = 0;
    msg->communication_round = 0;
    
    msg->stage1_weight_count = 0;
    msg->stage2_weight_count = 0;
    msg->stage1_weights = NULL;
    msg->stage2_weights = NULL;
    
    msg->local_loss = 0.0;
    msg->local_accuracy = 0.0;
    msg->processed_samples = 0;
    msg->timestamp = time(NULL);
    
    return msg;
}

void free_easgd_message(EASGDMessage* msg) {
    if (!msg) return;
    
    if (msg->stage1_weights) {
        free(msg->stage1_weights);
    }
    if (msg->stage2_weights) {
        free(msg->stage2_weights);
    }
    
    free(msg);
}

int send_easgd_message(int sockfd, EASGDMessage* msg) {
    if (!msg || sockfd < 0) return -1;
    
    // Send header first
    if (send(sockfd, msg, sizeof(EASGDMessage) - 2*sizeof(double*), 0) < 0) {
        perror("Failed to send EASGD message header");
        return -1;
    }
    
    // Send stage1 weights if present
    if (msg->stage1_weight_count > 0 && msg->stage1_weights) {
        size_t stage1_size = msg->stage1_weight_count * sizeof(double);
        if (send(sockfd, msg->stage1_weights, stage1_size, 0) < 0) {
            perror("Failed to send stage1 weights");
            return -1;
        }
    }
    
    // Send stage2 weights if present
    if (msg->stage2_weight_count > 0 && msg->stage2_weights) {
        size_t stage2_size = msg->stage2_weight_count * sizeof(double);
        if (send(sockfd, msg->stage2_weights, stage2_size, 0) < 0) {
            perror("Failed to send stage2 weights");
            return -1;
        }
    }
    
    return 0;
}

EASGDMessage* receive_easgd_message(int sockfd) {
    if (sockfd < 0) return NULL;
    
    EASGDMessage* msg = (EASGDMessage*)malloc(sizeof(EASGDMessage));
    if (!msg) return NULL;
    
    // Receive header
    ssize_t header_size = sizeof(EASGDMessage) - 2*sizeof(double*);
    if (recv(sockfd, msg, header_size, MSG_WAITALL) != header_size) {
        perror("Failed to receive EASGD message header");
        free(msg);
        return NULL;
    }
    
    msg->stage1_weights = NULL;
    msg->stage2_weights = NULL;
    
    // Receive stage1 weights if present
    if (msg->stage1_weight_count > 0) {
        size_t stage1_size = msg->stage1_weight_count * sizeof(double);
        msg->stage1_weights = (double*)malloc(stage1_size);
        if (!msg->stage1_weights) {
            free(msg);
            return NULL;
        }
        
        if (recv(sockfd, msg->stage1_weights, stage1_size, MSG_WAITALL) != (ssize_t)stage1_size) {
            perror("Failed to receive stage1 weights");
            free_easgd_message(msg);
            return NULL;
        }
    }
    
    // Receive stage2 weights if present
    if (msg->stage2_weight_count > 0) {
        size_t stage2_size = msg->stage2_weight_count * sizeof(double);
        msg->stage2_weights = (double*)malloc(stage2_size);
        if (!msg->stage2_weights) {
            free_easgd_message(msg);
            return NULL;
        }
        
        if (recv(sockfd, msg->stage2_weights, stage2_size, MSG_WAITALL) != (ssize_t)stage2_size) {
            perror("Failed to receive stage2 weights");
            free_easgd_message(msg);
            return NULL;
        }
    }
    
    return msg;
}

/* =============================================================================
 * EASGD ALGORITHM CORE IMPLEMENTATION
 * ============================================================================= */

int update_master_stage1_weights(HybridCoordinator* coord, 
                                double* worker1_weights, double* worker2_weights) {
    if (!coord || !worker1_weights || !worker2_weights) {
        printf("[EASGD] Error: Invalid parameters for Stage1 weight update\n");
        return -1;
    }
    
    pthread_mutex_lock(&coord->weights_mutex);
    
    printf("[EASGD] Updating master Stage1 weights (β=%.3f, count=%d)\n", 
           coord->beta, coord->stage1_weight_count);
    
    // EASGD update: w̄₁ ← w̄₁ + β(w₁₋₁ - w̄₁) + β(w₁₋₂ - w̄₁)
    for (int i = 0; i < coord->stage1_weight_count; i++) {
        double current_master = coord->master_stage1_weights[i];
        
        // Apply first worker's influence
        double diff1 = worker1_weights[i] - current_master;
        coord->master_stage1_weights[i] += coord->beta * diff1;
        
        // Apply second worker's influence
        double diff2 = worker2_weights[i] - coord->master_stage1_weights[i];
        coord->master_stage1_weights[i] += coord->beta * diff2;
    }
    
    pthread_mutex_unlock(&coord->weights_mutex);
    
    printf("[EASGD] Master Stage1 weights updated successfully\n");
    return 0;
}

int update_master_stage2_weights(HybridCoordinator* coord, double* worker_weights) {
    if (!coord || !worker_weights) {
        printf("[EASGD] Error: Invalid parameters for Stage2 weight update\n");
        return -1;
    }
    
    pthread_mutex_lock(&coord->weights_mutex);
    
    printf("[EASGD] Updating master Stage2 weights (β=%.3f, count=%d)\n", 
           coord->beta, coord->stage2_weight_count);
    
    // EASGD update: w̄₂ ← w̄₂ + β(w₂₋₁ - w̄₂)
    for (int i = 0; i < coord->stage2_weight_count; i++) {
        double diff = worker_weights[i] - coord->master_stage2_weights[i];
        coord->master_stage2_weights[i] += coord->beta * diff;
    }
    
    pthread_mutex_unlock(&coord->weights_mutex);
    
    printf("[EASGD] Master Stage2 weights updated successfully\n");
    return 0;
}

int apply_elastic_averaging_stage1(Stage1Worker* worker, double* master_weights) {
    if (!worker || !master_weights) {
        printf("[EASGD] Error: Invalid parameters for Stage1 elastic averaging\n");
        return -1;
    }
    
    printf("[STAGE1-%d] Applying elastic averaging (α₁=%.3f, count=%d)\n", 
           worker->worker_id, worker->alpha1, worker->stage1_weight_count);
    
    // Apply elastic averaging: w₁ ← w₁ + α₁(w̄₁ - w₁)
    for (int i = 0; i < worker->stage1_weight_count; i++) {
        double diff = master_weights[i] - worker->stage1_weights_flat[i];
        worker->stage1_weights_flat[i] += worker->alpha1 * diff;
    }
    
    // Reconstruct network weights from flattened array
    if (reconstruct_stage1_weights(worker, worker->stage1_weights_flat) < 0) {
        printf("[STAGE1-%d] Error: Failed to reconstruct Stage1 weights\n", worker->worker_id);
        return -1;
    }
    
    printf("[STAGE1-%d] Elastic averaging applied successfully\n", worker->worker_id);
    return 0;
}

int apply_elastic_averaging_stage2(Stage2Worker* worker, double* master_weights) {
    if (!worker || !master_weights) {
        printf("[EASGD] Error: Invalid parameters for Stage2 elastic averaging\n");
        return -1;
    }
    
    printf("[STAGE2-1] Applying elastic averaging (α₂=%.3f, count=%d)\n", 
           worker->alpha2, worker->stage2_weight_count);
    
    // Apply elastic averaging: w₂ ← w₂ + α₂(w̄₂ - w₂)
    for (int i = 0; i < worker->stage2_weight_count; i++) {
        double diff = master_weights[i] - worker->stage2_weights_flat[i];
        worker->stage2_weights_flat[i] += worker->alpha2 * diff;
    }
    
    // Reconstruct network weights from flattened array
    if (reconstruct_stage2_weights(worker, worker->stage2_weights_flat) < 0) {
        printf("[STAGE2-1] Error: Failed to reconstruct Stage2 weights\n");
        return -1;
    }
    
    printf("[STAGE2-1] Elastic averaging applied successfully\n");
    return 0;
}

/* =============================================================================
 * WEIGHT DIVERGENCE CALCULATION
 * ============================================================================= */

double calculate_stage1_weight_divergence(HybridCoordinator* coord, 
                                        EASGDMessage* worker1_msg, EASGDMessage* worker2_msg) {
    if (!coord || !worker1_msg || !worker2_msg) return -1.0;
    
    double total_divergence = 0.0;
    int count = coord->stage1_weight_count;
    
    pthread_mutex_lock(&coord->weights_mutex);
    
    for (int i = 0; i < count; i++) {
        double master_weight = coord->master_stage1_weights[i];
        double diff1 = worker1_msg->stage1_weights[i] - master_weight;
        double diff2 = worker2_msg->stage1_weights[i] - master_weight;
        
        total_divergence += (diff1 * diff1 + diff2 * diff2);
    }
    
    pthread_mutex_unlock(&coord->weights_mutex);
    
    double divergence = sqrt(total_divergence / count);
    coord->stage1_divergence = divergence;
    
    printf("[EASGD] Stage1 weight divergence: %.6f\n", divergence);
    return divergence;
}

double calculate_stage2_weight_divergence(HybridCoordinator* coord, EASGDMessage* worker_msg) {
    if (!coord || !worker_msg) return -1.0;
    
    double total_divergence = 0.0;
    int count = coord->stage2_weight_count;
    
    pthread_mutex_lock(&coord->weights_mutex);
    
    for (int i = 0; i < count; i++) {
        double master_weight = coord->master_stage2_weights[i];
        double diff = worker_msg->stage2_weights[i] - master_weight;
        
        total_divergence += (diff * diff);
    }
    
    pthread_mutex_unlock(&coord->weights_mutex);
    
    double divergence = sqrt(total_divergence / count);
    coord->stage2_divergence = divergence;
    
    printf("[EASGD] Stage2 weight divergence: %.6f\n", divergence);
    return divergence;
}

double calculate_total_system_divergence(HybridCoordinator* coord) {
    if (!coord) return -1.0;
    
    double total_divergence = coord->stage1_divergence + coord->stage2_divergence;
    coord->total_weight_divergence = total_divergence;
    
    printf("[EASGD] Total system weight divergence: %.6f\n", total_divergence);
    return total_divergence;
}

/* =============================================================================
 * CONVERGENCE DETECTION
 * ============================================================================= */

int detect_convergence(HybridCoordinator* coord, double threshold) {
    if (!coord) return -1;
    
    double current_divergence = calculate_total_system_divergence(coord);
    
    if (current_divergence < threshold) {
        printf("[EASGD] Convergence detected at divergence %.6f (threshold: %.6f)\n", 
               current_divergence, threshold);
        return 1;
    }
    
    return 0;
}

void track_convergence_metrics(HybridCoordinator* coord, int epoch) {
    if (!coord) return;
    
    // Calculate convergence rates
    static double prev_stage1_divergence = 1.0;
    static double prev_stage2_divergence = 1.0;
    
    if (epoch > 0) {
        coord->aggregated_stats.convergence_rate = 
            (prev_stage1_divergence + prev_stage2_divergence - 
             coord->stage1_divergence - coord->stage2_divergence) / 
            (prev_stage1_divergence + prev_stage2_divergence);
    }
    
    prev_stage1_divergence = coord->stage1_divergence;
    prev_stage2_divergence = coord->stage2_divergence;
    
    printf("[EASGD] Epoch %d convergence rate: %.4f\n", 
           epoch, coord->aggregated_stats.convergence_rate);
}

/* =============================================================================
 * EASGD PERFORMANCE OPTIMIZATION
 * ============================================================================= */

void set_optimal_hybrid_parameters(double* alpha1, double* alpha2, double* beta, 
                                  const char* network_type) {
    if (!alpha1 || !alpha2 || !beta || !network_type) return;
    
    if (strcmp(network_type, "LAN") == 0) {
        // LAN: Higher communication frequency
        *alpha1 = 0.3;
        *alpha2 = 0.4;
        *beta = 0.6;
        printf("[EASGD] LAN parameters: α₁=%.1f, α₂=%.1f, β=%.1f\n", *alpha1, *alpha2, *beta);
    } else if (strcmp(network_type, "WAN") == 0) {
        // WAN: Lower communication frequency
        *alpha1 = 0.1;
        *alpha2 = 0.2;
        *beta = 0.3;
        printf("[EASGD] WAN parameters: α₁=%.1f, α₂=%.1f, β=%.1f\n", *alpha1, *alpha2, *beta);
    } else {
        // LOCAL: Balanced parameters
        *alpha1 = DEFAULT_ALPHA_STAGE1;
        *alpha2 = DEFAULT_ALPHA_STAGE2;
        *beta = DEFAULT_BETA;
        printf("[EASGD] LOCAL parameters: α₁=%.1f, α₂=%.1f, β=%.1f\n", *alpha1, *alpha2, *beta);
    }
}

int validate_hybrid_configuration(int num_stage1_workers, int num_stage2_workers) {
    if (num_stage1_workers != 2) {
        printf("[EASGD] Error: Expected 2 Stage1 workers, got %d\n", num_stage1_workers);
        return -1;
    }
    
    if (num_stage2_workers != 1) {
        printf("[EASGD] Error: Expected 1 Stage2 worker, got %d\n", num_stage2_workers);
        return -1;
    }
    
    printf("[EASGD] Hybrid configuration validated: %d Stage1 + %d Stage2 workers\n", 
           num_stage1_workers, num_stage2_workers);
    return 0;
}

/* =============================================================================
 * DEBUGGING AND MONITORING
 * ============================================================================= */

void log_easgd_synchronization(int round, double divergence, double comm_time) {
    printf("EASGD_LOG|%ld|SYNC|round_%d|divergence_%.6f|comm_time_%.3f\n", 
           time(NULL), round, divergence, comm_time);
    fflush(stdout);
}

void debug_weight_consistency(HybridCoordinator* coord) {
    if (!coord) return;
    
    printf("[DEBUG] Master weights consistency check:\n");
    printf("        Stage1 weight count: %d\n", coord->stage1_weight_count);
    printf("        Stage2 weight count: %d\n", coord->stage2_weight_count);
    
    if (coord->master_stage1_weights && coord->stage1_weight_count > 0) {
        double sum = 0.0;
        for (int i = 0; i < min(10, coord->stage1_weight_count); i++) {
            sum += coord->master_stage1_weights[i];
        }
        printf("        Stage1 first 10 weights sum: %.6f\n", sum);
    }
    
    if (coord->master_stage2_weights && coord->stage2_weight_count > 0) {
        double sum = 0.0;
        for (int i = 0; i < min(10, coord->stage2_weight_count); i++) {
            sum += coord->master_stage2_weights[i];
        }
        printf("        Stage2 first 10 weights sum: %.6f\n", sum);
    }
}