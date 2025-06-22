#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdarg.h> 
#include <unistd.h>
#include <math.h>  // THÊM cho exp, log functions
#include "../neural/nn.h"
#include "../socket/socket_utils.h"

// THÊM: Structure để track worker updates
typedef struct {
    int worker_id;
    double loss;
    double* weights;
    int weight_count;
    double timestamp;
} WorkerUpdate;

// NEW: Global loss statistics for comparison
typedef struct {
    double loss_sum;
    double loss_min;
    double loss_max;
    int update_count;
    double recent_losses[50];  // Sliding window
    int recent_index;
    double moving_avg;
} GlobalLossStats;

GlobalLossStats global_stats = {0.0, 1e9, -1e9, 0, {0}, 0, 0.0};

// NEW: Get global relative weight based on loss performance
double get_global_relative_weight(double worker_loss) {
    // Update global stats
    global_stats.loss_sum += worker_loss;
    global_stats.update_count++;
    
    // Update min/max
    if (worker_loss < global_stats.loss_min) global_stats.loss_min = worker_loss;
    if (worker_loss > global_stats.loss_max) global_stats.loss_max = worker_loss;
    
    // Update sliding window
    global_stats.recent_losses[global_stats.recent_index] = worker_loss;
    global_stats.recent_index = (global_stats.recent_index + 1) % 50;
    
    // Compute moving average from recent losses
    double recent_sum = 0.0;
    int window_size = (global_stats.update_count < 50) ? global_stats.update_count : 50;
    for (int i = 0; i < window_size; i++) {
        recent_sum += global_stats.recent_losses[i];
    }
    global_stats.moving_avg = recent_sum / window_size;
    
    // Early stage: use uniform weighting until we have enough data
    if (global_stats.update_count < 5) {
        printf("[PS] Early stage: uniform weight=1.0 (updates=%d)\n", global_stats.update_count);
        return 1.0;
    }
    
    // Method 1: Relative to moving average
    double avg_ratio = global_stats.moving_avg / (worker_loss + 1e-8);
    double avg_weight = 0.5 + 0.5 * fmin(2.0, avg_ratio);  // [0.5, 1.5]
    
    // Method 2: Relative to min-max range
    double loss_range = global_stats.loss_max - global_stats.loss_min + 1e-8;
    double normalized_loss = (worker_loss - global_stats.loss_min) / loss_range;
    double range_weight = 1.5 - normalized_loss;  // Best → 1.5, worst → 0.5
    
    // Method 3: Exponential weighting based on performance
    double exp_weight = 0.8 + 0.4 * exp(-worker_loss);  // [0.8, 1.2]
    
    // Combine methods with weights
    double combined_weight = 0.4 * avg_weight + 0.4 * range_weight + 0.2 * exp_weight;
    
    // Clamp to reasonable range
    combined_weight = fmax(0.3, fmin(1.7, combined_weight));
    
    printf("[PS] Global stats: avg=%.6f, range=[%.6f,%.6f], moving_avg=%.6f\n",
           global_stats.loss_sum / global_stats.update_count, 
           global_stats.loss_min, global_stats.loss_max, global_stats.moving_avg);
    printf("[PS] Worker loss=%.6f → weights: avg=%.3f, range=%.3f, exp=%.3f → final=%.3f\n",
           worker_loss, avg_weight, range_weight, exp_weight, combined_weight);
    
    return combined_weight;
}

// NEW: Adaptive beta based on loss quality
double get_adaptive_beta(double worker_loss, double base_beta) {
    // Good loss threshold (adjust based on your data)
    double excellent_loss = 0.05;  // Very good performance
    double good_loss = 0.1;        // Acceptable performance
    double poor_loss = 0.5;        // Poor performance
    
    double beta_multiplier;
    
    if (worker_loss < excellent_loss) {
        // Excellent loss → increase learning rate significantly
        beta_multiplier = 2.0 + 1.0 * exp(-worker_loss / excellent_loss);
    } else if (worker_loss < good_loss) {
        // Good loss → moderate increase
        beta_multiplier = 1.2 + 0.8 * (good_loss - worker_loss) / good_loss;
    } else if (worker_loss < poor_loss) {
        // Average loss → normal rate
        beta_multiplier = 0.8 + 0.4 * (poor_loss - worker_loss) / (poor_loss - good_loss);
    } else {
        // Poor loss → decrease learning rate
        beta_multiplier = 0.5 * exp(-(worker_loss - poor_loss));
    }
    
    // Clamp multiplier
    beta_multiplier = fmax(0.1, fmin(3.0, beta_multiplier));
    
    double adaptive_beta = base_beta * beta_multiplier;
    
    printf("[PS] Loss quality assessment: %.6f → β: %.6f → %.6f (%.1fx)\n",
           worker_loss, base_beta, adaptive_beta, beta_multiplier);
    
    return adaptive_beta;
}

// MODIFIED: Original function for backward compatibility
double* compute_loss_weights(WorkerUpdate* updates, int num_updates) {
    double* weights = malloc(sizeof(double) * num_updates);
    double sum_inv_loss = 0.0;
    double epsilon = 1e-8;  // Numerical stability
    
    printf("[Parameter Server] Computing loss weights for %d workers:\n", num_updates);
    
    // Compute 1/Lᵢ with numerical stability
    for (int i = 0; i < num_updates; i++) {
        double inv_loss = 1.0 / (updates[i].loss + epsilon);
        weights[i] = inv_loss;
        sum_inv_loss += inv_loss;
        printf("[Parameter Server]   Worker %d: loss=%.6f -> inv_weight=%.3f\n", 
               updates[i].worker_id, updates[i].loss, inv_loss);
    }
    
    // Normalize: wᵢ = (1/Lᵢ) / Σ(1/Lⱼ)
    printf("[Parameter Server] Normalized weights: ");
    for (int i = 0; i < num_updates; i++) {
        weights[i] /= sum_inv_loss;
        printf("w%d=%.3f ", updates[i].worker_id, weights[i]);
    }
    printf("\n");
    fflush(stdout);
    
    return weights;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <port>\n", argv[0]);
        return 1;
    }
    
    int port = atoi(argv[1]);
    printf("[Parameter Server] Starting GLOBAL-LOSS-ADAPTIVE EASGD variant on port %d\n", port);
    fflush(stdout);
    
    srand(time(NULL));
    
    // Initialize template network và elastic center
    NeuralNetwork* template_net = network_create(784, 300, 10, 0.1);
    network_easgd_init(template_net, 0.0, 0.001);  // α=0.0 (no local elastic), β=0.001
    elastic_center_init(template_net);
    
    printf("[Parameter Server] Elastic center initialized (β=0.001, global-loss-adaptive)\n");
    fflush(stdout);
    
    // Setup server socket
    int server_sock = setup_server(port);
    printf("[Parameter Server] Server socket ready, waiting for workers...\n");
    fflush(stdout);
    
    int request_count = 0;
    int save_interval = 60;
    
    while (1) {
        printf("[Parameter Server] Waiting for worker connection...\n");
        fflush(stdout);
        
        int client_sock = accept_client(server_sock);
        request_count++;
        
        double request_start = time_in_socket_seconds();
        double comm_start_1, comm_end_1;
        double comm_start_2, comm_end_2;
        
        printf("[Parameter Server] Worker connected (request #%d), starting timer\n", request_count);
        fflush(stdout);
        
        // Receive worker ID
        int worker_id;
        if (recv_all(client_sock, &worker_id, sizeof(int)) != sizeof(int)) {
            printf("[Parameter Server] Failed to receive worker ID\n");
            close(client_sock);
            continue;
        }
        
        // THÊM: Receive worker loss
        double worker_loss;
        if (recv_all(client_sock, &worker_loss, sizeof(double)) != sizeof(double)) {
            printf("[Parameter Server] Failed to receive worker loss\n");
            close(client_sock);
            continue;
        }
        
        // Receive weight count
        int weight_count;
        if (recv_all(client_sock, &weight_count, sizeof(int)) != sizeof(int)) {
            printf("[Parameter Server] Failed to receive weight count\n");
            close(client_sock);
            continue;
        }

        // === TIMING: Receive worker weights ===
        comm_start_1 = time_in_socket_seconds();
        double* worker_weights = malloc(sizeof(double) * weight_count);
        if (recv_all(client_sock, worker_weights, sizeof(double) * weight_count) != sizeof(double) * weight_count) {
            printf("[Parameter Server] Failed to receive weights\n");
            free(worker_weights);
            close(client_sock);
            continue;
        }
        comm_end_1 = time_in_socket_seconds();
        
        printf("[Parameter Server] Received %d weights from worker %d (loss=%.6f) in %.3fms\n", 
               weight_count, worker_id, worker_loss, (comm_end_1 - comm_start_1) * 1000);
        fflush(stdout);
        
        // NEW: Global loss comparison weighting
        double update_start = time_in_socket_seconds();
        
        // METHOD 1: Global relative weighting
        double global_weight = get_global_relative_weight(worker_loss);
        
        // METHOD 2: Adaptive beta
        double adaptive_beta = get_adaptive_beta(worker_loss, template_net->beta);
        
        // Choose between methods (you can experiment)
        #define USE_GLOBAL_WEIGHT 1  // Set to 0 to use adaptive beta instead
        
        #if USE_GLOBAL_WEIGHT
        // Use global weight with original beta
        elastic_center_weighted_update(template_net, worker_weights, weight_count, global_weight);
        printf("[Parameter Server] Applied GLOBAL WEIGHT update: weight=%.3f\n", global_weight);
        #else
        // Use adaptive beta with weight=1.0
        // Need to modify elastic_center_weighted_update to accept custom beta
        elastic_center_adaptive_beta_update(template_net, worker_weights, weight_count, 1.0, adaptive_beta);
        printf("[Parameter Server] Applied ADAPTIVE BETA update: β=%.6f\n", adaptive_beta);
        #endif
        
        double update_end = time_in_socket_seconds();
        printf("[Parameter Server] Updated elastic center in %.3fms\n", 
               (update_end - update_start) * 1000);
        fflush(stdout);
        
        // Get center weights
        int center_count;
        double* center_weights = elastic_center_get_weights(&center_count);
        
        // === TIMING: Send center weights ===
        comm_start_2 = time_in_socket_seconds();
        if (send_all(client_sock, &center_count, sizeof(int)) == sizeof(int) &&
            send_all(client_sock, center_weights, sizeof(double) * center_count) == sizeof(double) * center_count) {
            comm_end_2 = time_in_socket_seconds();
            printf("[Parameter Server] Sent %d center weights to worker %d in %.3fms\n", 
                   center_count, worker_id, (comm_end_2 - comm_start_2) * 1000);
        } else {
            printf("[Parameter Server] Failed to send center to worker %d\n", worker_id);
        }
        fflush(stdout);

        double request_end = time_in_socket_seconds();
        
        // Performance metrics
        double total_processing_time = (request_end - request_start) * 1000;
        double recv_time = (comm_end_1 - comm_start_1) * 1000;
        double update_time = (update_end - update_start) * 1000;
        double send_time = (comm_end_2 - comm_start_2) * 1000;
        double total_comm_time = recv_time + send_time;
        
        printf("[Parameter Server] Request #%d: %.3fms total (recv=%.3fms, adaptive_update=%.3fms, send=%.3fms)\n", 
               request_count, total_processing_time, recv_time, update_time, send_time);
        printf("[Parameter Server] Communication: %.1f%%, Computation: %.1f%% (global-adaptive)\n",
               (total_comm_time / total_processing_time) * 100,
               (update_time / total_processing_time) * 100);
        fflush(stdout);

        // Cleanup
        free(worker_weights);
        free(center_weights);
        close(client_sock);
        
        // Periodic save
        if (request_count % save_interval == 0) {
            printf("[Parameter Server] Saving elastic center after %d requests...\n", request_count);
            fflush(stdout);
            
            int save_center_count;
            double* save_center_weights = elastic_center_get_weights(&save_center_count);
            network_set_weights(template_net, save_center_weights, save_center_count);
            
            char model_name[256];
            sprintf(model_name, "elastic_center_checkpoint_%d", request_count);
            network_save(template_net, model_name);
            printf("[Parameter Server] Checkpoint saved to %s\n", model_name);
            fflush(stdout);
            
            free(save_center_weights);
        }
        
        if (request_count % 10 == 0) {
            printf("[Parameter Server] === Processed %d total requests ===\n", request_count);
            printf("[Parameter Server] Global Loss Stats: updates=%d, avg=%.6f, range=[%.6f,%.6f]\n",
                   global_stats.update_count, 
                   global_stats.loss_sum / global_stats.update_count,
                   global_stats.loss_min, global_stats.loss_max);
            fflush(stdout);
        }
    }
    
    return 0;
}