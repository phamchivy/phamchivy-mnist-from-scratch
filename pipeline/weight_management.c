/**
 * @file weight_management.c
 * @brief Weight Flattening and Reconstruction for EASGD Communication
 * @author Research Team
 * @date 2024
 * 
 * This file implements weight serialization/deserialization functions for
 * EASGD communication between workers and parameter server. It converts
 * 2D neural network weight matrices into 1D arrays for network transmission
 * and reconstructs them back to matrix form.
 */

#include "hybrid_pipeline_easgd.h"
#include <string.h>
#include <stdlib.h>      // malloc, free, getenv
#include <stdio.h>       // printf, fprintf, fopen, fclose, FILE, stdout, fflush
#include <string.h>      // memset, strcpy, strcmp
#include <netinet/in.h>  // struct sockaddr_in, socklen_t
#include <sys/socket.h>  // accept

/* =============================================================================
 * STAGE1 WEIGHT MANAGEMENT
 * ============================================================================= */

int flatten_stage1_weights(Stage1Worker* worker) {
    if (!worker || !worker->stage1) {
        printf("[WEIGHT_MGMT] Error: Invalid Stage1 worker for weight flattening\n");
        return -1;
    }
    
    NetworkStage* stage1 = worker->stage1;
    
    // Calculate total parameter count
    int weight_params = stage1->weights->rows * stage1->weights->cols;
    int bias_params = stage1->biases->rows * stage1->biases->cols;
    int total_params = weight_params + bias_params;
    
    // Allocate flattened array if needed
    if (!worker->stage1_weights_flat) {
        worker->stage1_weights_flat = (double*)malloc(sizeof(double) * total_params);
        if (!worker->stage1_weights_flat) {
            printf("[WEIGHT_MGMT] Error: Failed to allocate Stage1 weight array\n");
            return -1;
        }
    }
    
    int idx = 0;
    
    // Flatten weight matrix (row-major order)
    for (int i = 0; i < stage1->weights->rows; i++) {
        for (int j = 0; j < stage1->weights->cols; j++) {
            worker->stage1_weights_flat[idx++] = stage1->weights->entries[i][j];
        }
    }
    
    // Flatten bias vector
    for (int i = 0; i < stage1->biases->rows; i++) {
        for (int j = 0; j < stage1->biases->cols; j++) {
            worker->stage1_weights_flat[idx++] = stage1->biases->entries[i][j];
        }
    }
    
    worker->stage1_weight_count = total_params;
    
    printf("[STAGE1-%d] Flattened %d weights (%d matrix + %d bias)\n", 
           worker->worker_id, total_params, weight_params, bias_params);
    
    return total_params;
}

int reconstruct_stage1_weights(Stage1Worker* worker, double* flat_weights) {
    if (!worker || !worker->stage1 || !flat_weights) {
        printf("[WEIGHT_MGMT] Error: Invalid parameters for Stage1 weight reconstruction\n");
        return -1;
    }
    
    NetworkStage* stage1 = worker->stage1;
    int idx = 0;
    
    // Reconstruct weight matrix
    for (int i = 0; i < stage1->weights->rows; i++) {
        for (int j = 0; j < stage1->weights->cols; j++) {
            stage1->weights->entries[i][j] = flat_weights[idx++];
        }
    }
    
    // Reconstruct bias vector
    for (int i = 0; i < stage1->biases->rows; i++) {
        for (int j = 0; j < stage1->biases->cols; j++) {
            stage1->biases->entries[i][j] = flat_weights[idx++];
        }
    }
    
    printf("[STAGE1-%d] Reconstructed Stage1 weights from flat array\n", worker->worker_id);
    return 0;
}

/* =============================================================================
 * STAGE2 WEIGHT MANAGEMENT
 * ============================================================================= */

int flatten_stage2_weights(Stage2Worker* worker) {
    if (!worker || !worker->stage2) {
        printf("[WEIGHT_MGMT] Error: Invalid Stage2 worker for weight flattening\n");
        return -1;
    }
    
    NetworkStage* stage2 = worker->stage2;
    
    // Calculate total parameter count
    int weight_params = stage2->weights->rows * stage2->weights->cols;
    int bias_params = stage2->biases->rows * stage2->biases->cols;
    int total_params = weight_params + bias_params;
    
    // Allocate flattened array if needed
    if (!worker->stage2_weights_flat) {
        worker->stage2_weights_flat = (double*)malloc(sizeof(double) * total_params);
        if (!worker->stage2_weights_flat) {
            printf("[WEIGHT_MGMT] Error: Failed to allocate Stage2 weight array\n");
            return -1;
        }
    }
    
    int idx = 0;
    
    // Flatten weight matrix (row-major order)
    for (int i = 0; i < stage2->weights->rows; i++) {
        for (int j = 0; j < stage2->weights->cols; j++) {
            worker->stage2_weights_flat[idx++] = stage2->weights->entries[i][j];
        }
    }
    
    // Flatten bias vector
    for (int i = 0; i < stage2->biases->rows; i++) {
        for (int j = 0; j < stage2->biases->cols; j++) {
            worker->stage2_weights_flat[idx++] = stage2->biases->entries[i][j];
        }
    }
    
    worker->stage2_weight_count = total_params;
    
    printf("[STAGE2-1] Flattened %d weights (%d matrix + %d bias)\n", 
           total_params, weight_params, bias_params);
    
    return total_params;
}

int reconstruct_stage2_weights(Stage2Worker* worker, double* flat_weights) {
    if (!worker || !worker->stage2 || !flat_weights) {
        printf("[WEIGHT_MGMT] Error: Invalid parameters for Stage2 weight reconstruction\n");
        return -1;
    }
    
    NetworkStage* stage2 = worker->stage2;
    int idx = 0;
    
    // Reconstruct weight matrix
    for (int i = 0; i < stage2->weights->rows; i++) {
        for (int j = 0; j < stage2->weights->cols; j++) {
            stage2->weights->entries[i][j] = flat_weights[idx++];
        }
    }
    
    // Reconstruct bias vector
    for (int i = 0; i < stage2->biases->rows; i++) {
        for (int j = 0; j < stage2->biases->cols; j++) {
            stage2->biases->entries[i][j] = flat_weights[idx++];
        }
    }
    
    printf("[STAGE2-1] Reconstructed Stage2 weights from flat array\n");
    return 0;
}

/* =============================================================================
 * COORDINATOR WEIGHT MANAGEMENT
 * ============================================================================= */

int initialize_master_weights(HybridCoordinator* coord) {
    if (!coord) {
        printf("[WEIGHT_MGMT] Error: Invalid coordinator for weight initialization\n");
        return -1;
    }
    
    // Calculate weight counts for standard MNIST network
    // Stage1: 784 -> 512 (weights: 784*512 + bias: 512)
    coord->stage1_weight_count = 784 * 512 + 512;
    
    // Stage2: 512 -> 10 (weights: 512*10 + bias: 10)
    coord->stage2_weight_count = 512 * 10 + 10;
    
    // Allocate master weight arrays
    coord->master_stage1_weights = (double*)calloc(coord->stage1_weight_count, sizeof(double));
    coord->master_stage2_weights = (double*)calloc(coord->stage2_weight_count, sizeof(double));
    
    if (!coord->master_stage1_weights || !coord->master_stage2_weights) {
        printf("[WEIGHT_MGMT] Error: Failed to allocate master weight arrays\n");
        if (coord->master_stage1_weights) free(coord->master_stage1_weights);
        if (coord->master_stage2_weights) free(coord->master_stage2_weights);
        return -1;
    }
    
    // Initialize with small random values (Xavier initialization)
    srand(time(NULL));
    
    // Initialize Stage1 master weights
    double stage1_std = sqrt(2.0 / (784 + 512));
    for (int i = 0; i < coord->stage1_weight_count; i++) {
        coord->master_stage1_weights[i] = ((double)rand() / RAND_MAX - 0.5) * 2.0 * stage1_std;
    }
    
    // Initialize Stage2 master weights
    double stage2_std = sqrt(2.0 / (512 + 10));
    for (int i = 0; i < coord->stage2_weight_count; i++) {
        coord->master_stage2_weights[i] = ((double)rand() / RAND_MAX - 0.5) * 2.0 * stage2_std;
    }
    
    printf("[COORD] Initialized master weights: Stage1=%d params, Stage2=%d params\n",
           coord->stage1_weight_count, coord->stage2_weight_count);
    
    return 0;
}

/* =============================================================================
 * WEIGHT COPY AND VALIDATION FUNCTIONS
 * ============================================================================= */

int copy_master_stage1_weights(HybridCoordinator* coord, double* destination) {
    if (!coord || !destination || !coord->master_stage1_weights) {
        printf("[WEIGHT_MGMT] Error: Invalid parameters for Stage1 weight copy\n");
        return -1;
    }
    
    pthread_mutex_lock(&coord->weights_mutex);
    memcpy(destination, coord->master_stage1_weights, 
           coord->stage1_weight_count * sizeof(double));
    pthread_mutex_unlock(&coord->weights_mutex);
    
    return coord->stage1_weight_count;
}

int copy_master_stage2_weights(HybridCoordinator* coord, double* destination) {
    if (!coord || !destination || !coord->master_stage2_weights) {
        printf("[WEIGHT_MGMT] Error: Invalid parameters for Stage2 weight copy\n");
        return -1;
    }
    
    pthread_mutex_lock(&coord->weights_mutex);
    memcpy(destination, coord->master_stage2_weights, 
           coord->stage2_weight_count * sizeof(double));
    pthread_mutex_unlock(&coord->weights_mutex);
    
    return coord->stage2_weight_count;
}

int validate_weight_consistency(HybridCoordinator* coord) {
    if (!coord) return -1;
    
    int errors = 0;
    
    // Check for NaN or infinite values in master weights
    for (int i = 0; i < coord->stage1_weight_count; i++) {
        if (isnan(coord->master_stage1_weights[i]) || isinf(coord->master_stage1_weights[i])) {
            printf("[WEIGHT_MGMT] Warning: Invalid Stage1 master weight at index %d: %f\n", 
                   i, coord->master_stage1_weights[i]);
            errors++;
        }
    }
    
    for (int i = 0; i < coord->stage2_weight_count; i++) {
        if (isnan(coord->master_stage2_weights[i]) || isinf(coord->master_stage2_weights[i])) {
            printf("[WEIGHT_MGMT] Warning: Invalid Stage2 master weight at index %d: %f\n", 
                   i, coord->master_stage2_weights[i]);
            errors++;
        }
    }
    
    if (errors > 0) {
        printf("[WEIGHT_MGMT] Weight validation failed: %d invalid values found\n", errors);
        return -1;
    }
    
    printf("[WEIGHT_MGMT] Weight validation passed: All values are valid\n");
    return 0;
}

/* =============================================================================
 * DATASET PARTITIONING UTILITIES
 * ============================================================================= */

Img** load_dataset_partition(int start_idx, int end_idx, const char* dataset_path) {
    if (start_idx < 0 || end_idx <= start_idx || !dataset_path) {
        printf("[WEIGHT_MGMT] Error: Invalid dataset partition parameters\n");
        return NULL;
    }
    
    printf("[WEIGHT_MGMT] Loading dataset partition [%d:%d] from %s\n", 
           start_idx, end_idx, dataset_path);
    
    // Load full dataset first
    int total_images = 60000; // MNIST training set size
    Img** full_dataset = csv_to_imgs(dataset_path, total_images);
    if (!full_dataset) {
        printf("[WEIGHT_MGMT] Error: Failed to load full dataset\n");
        return NULL;
    }
    
    // Create partition array
    int partition_size = end_idx - start_idx;
    Img** partition = (Img**)malloc(sizeof(Img*) * partition_size);
    if (!partition) {
        printf("[WEIGHT_MGMT] Error: Failed to allocate partition array\n");
        imgs_free(full_dataset, total_images);
        return NULL;
    }
    
    // Copy partition images (transfer ownership)
    for (int i = 0; i < partition_size; i++) {
        if (start_idx + i < total_images && full_dataset[start_idx + i]) {
            partition[i] = full_dataset[start_idx + i];
            full_dataset[start_idx + i] = NULL; // Transfer ownership
        } else {
            partition[i] = NULL;
        }
    }
    
    // Clean up remaining images in full dataset
    for (int i = 0; i < total_images; i++) {
        if (full_dataset[i]) {
            img_free(full_dataset[i]);
        }
    }
    free(full_dataset);
    
    printf("[WEIGHT_MGMT] Successfully loaded %d images for partition [%d:%d]\n", 
           partition_size, start_idx, end_idx);
    
    return partition;
}

/* =============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================= */

int calculate_total_stage1_parameters(NetworkStage* stage1) {
    if (!stage1 || !stage1->weights || !stage1->biases) return -1;
    
    int weight_params = stage1->weights->rows * stage1->weights->cols;
    int bias_params = stage1->biases->rows * stage1->biases->cols;
    
    return weight_params + bias_params;
}

int calculate_total_stage2_parameters(NetworkStage* stage2) {
    if (!stage2 || !stage2->weights || !stage2->biases) return -1;
    
    int weight_params = stage2->weights->rows * stage2->weights->cols;
    int bias_params = stage2->biases->rows * stage2->biases->cols;
    
    return weight_params + bias_params;
}

void print_weight_statistics(double* weights, int count, const char* name) {
    if (!weights || count <= 0 || !name) return;
    
    double sum = 0.0, min_val = weights[0], max_val = weights[0];
    
    for (int i = 0; i < count; i++) {
        sum += weights[i];
        if (weights[i] < min_val) min_val = weights[i];
        if (weights[i] > max_val) max_val = weights[i];
    }
    
    double mean = sum / count;
    
    // Calculate standard deviation
    double variance = 0.0;
    for (int i = 0; i < count; i++) {
        double diff = weights[i] - mean;
        variance += diff * diff;
    }
    double std_dev = sqrt(variance / count);
    
    printf("[WEIGHT_STATS] %s: count=%d, mean=%.6f, std=%.6f, min=%.6f, max=%.6f\n",
           name, count, mean, std_dev, min_val, max_val);
}