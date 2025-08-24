#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "../neural/nn.h"
#include "../util/img.h"
#include "../socket/socket_utils.h"
#include "../socket/pipeline_socket.h"
#include "../config/config_loader.h"

int main(int argc, char** argv) {
    // if (argc < 6) {
    //     printf("Usage: %s <group_id> <listen_port> <server_ip> <server_port>\n", argv[0]);
    //     return 1;
    // }

    if (argc < 5) {
        printf("Usage: %s <config_file> <group_id> <listen_port> <server_ip> <server_port>\n", argv[0]);
        printf("   OR: %s <group_id> <listen_port> <server_ip> <server_port> (uses default config.yml)\n", argv[0]);
        return 1;
    }

    // Load configuration
    HybridConfig* config;
    int group_id, listen_port, server_port;
    char server_ip[64];
    
    if (argc == 5) {
        // Old format - use default config
        config = load_config("config.yml");
        group_id = atoi(argv[1]);
        listen_port = atoi(argv[2]);
        strcpy(server_ip, argv[3]);
        server_port = atoi(argv[4]);
    } else {
        // New format - config file specified
        config = load_config(argv[1]);
        group_id = atoi(argv[2]);
        listen_port = atoi(argv[3]);
        strcpy(server_ip, argv[4]);
        server_port = atoi(argv[5]);
    }
    
    if (!config) {
        printf("Failed to load configuration\n");
        return 1;
    }
    
    // int group_id = atoi(argv[2]);
    // int listen_port = atoi(argv[3]);
    // const char* server_ip = argv[4];
    // int server_port = atoi(argv[5]);
    
    // printf("[Stage2 Group %d] Starting, listening on port %d, server: %s:%d\n", 
    //        group_id, listen_port, server_ip, server_port);
    // fflush(stdout);

    printf("[Stage2 Group %d] Starting with config-driven parameters\n", group_id);
    printf("[Stage2 Group %d] EASGD: α=%.3f, β=%.3f, sync_freq=%d\n", 
           group_id, config->easgd.alpha, config->easgd.beta, config->pipeline.sync_frequency);
    printf("[Stage2 Group %d] Listening on port %d, server: %s:%d\n", 
           group_id, listen_port, server_ip, server_port);
    fflush(stdout);
    
    srand(time(NULL) + group_id + 100);
    
    // Initialize pipeline stage 2 (300 → 10)
    // PipelineStage* stage2 = pipeline_stage_create(300, 10, 0.1);
    // stage2->easgd_enabled = true;
    // stage2->alpha = 0.01;
    // stage2->beta = 0.001;
    
    // printf("[Stage2 Group %d] Pipeline stage initialized: 300→10, α=0.01, β=0.001\n", group_id);
    // fflush(stdout);

    // Initialize pipeline stage 2 using config parameters
    PipelineStage* stage2 = pipeline_stage_create(config->network.hidden_size, 
                                                 config->network.output_size, 
                                                 config->training.learning_rate);
    stage2->easgd_enabled = config->easgd.enabled;
    stage2->alpha = config->easgd.alpha;
    stage2->beta = config->easgd.beta;

    printf("[Stage2 Group %d] Pipeline stage initialized: %d→%d, lr=%.3f, α=%.3f, β=%.3f\n", 
           group_id, config->network.hidden_size, config->network.output_size,
           config->training.learning_rate, config->easgd.alpha, config->easgd.beta);
    fflush(stdout);
    
    // Setup server to receive activations from stage 1
    int server_sock = setup_activation_server(listen_port);
    if (server_sock < 0) {
        printf("[Stage2 Group %d] Failed to setup activation server\n", group_id);
        free_config(config);
        return 1;
    }
    
    printf("[Stage2 Group %d] Activation server ready on port %d\n", group_id, listen_port);
    fflush(stdout);
    
    int processed_count = 0;
    int sync_count = 0;
    double loss_sum = 0.0;
    double start_time = time_in_socket_seconds();

    // Calculate target images per group
    int images_per_group = config->training.total_images / config->pipeline.num_groups;
    
    while (processed_count < images_per_group) {
        // Accept connection from stage 1
        int stage1_sock = accept_client(server_sock);
        if (stage1_sock < 0) continue;
        
        // Receive image label
        int label;
        if (recv_all(stage1_sock, &label, sizeof(int)) != sizeof(int)) {
            close(stage1_sock);
            continue;
        }
        
        // Receive hidden activation from stage 1
        Matrix* hidden_activation = receive_activation(stage1_sock, config->network.hidden_size, 1);
        if (hidden_activation == NULL) {
            close(stage1_sock);
            continue;
        }
        
        // Create target output
        Matrix* target = matrix_create(config->network.output_size, 1);
        target->entries[label][0] = 1.0;
        Matrix* final_outputs = pipeline_stage2_forward(stage2, hidden_activation);
        
        // Forward and backward pass through stage 2
        Matrix* grad_to_stage1;
        double loss = pipeline_stage2_backward(stage2, hidden_activation, final_outputs, 
                                            target, &grad_to_stage1);
        loss_sum += loss;
        processed_count++;
        
        // Send gradient back to stage 1
        send_gradient(stage1_sock, grad_to_stage1);
        close(stage1_sock);
        
        // Cleanup
        matrix_free(hidden_activation);
        matrix_free(target);
        matrix_free(grad_to_stage1);
        
        // Sync every 1000 images
        if (processed_count % config->pipeline.sync_frequency == 0) {
            double avg_loss = loss_sum / config->pipeline.sync_frequency;
            
            if (config->logging.log_loss) {
                printf("[Stage2 Group %d] Syncing after %d images (sync #%d), avg_loss=%.6f\n", 
                       group_id, processed_count, ++sync_count, avg_loss);
            } else {
                printf("[Stage2 Group %d] Syncing after %d images (sync #%d)\n", 
                       group_id, processed_count, ++sync_count);
            }
            
            // Get stage 2 weights
            int weight_count;
            double* weights = pipeline_stage2_get_weights(stage2, &weight_count);
            
            // Send to parameter server
            if (send_weights_to_parameter_server(server_ip, server_port, group_id, 
                                               WEIGHT_TYPE_OUTPUT, weights, weight_count) == 0) {
                if (config->logging.log_sync_details) {
                    printf("[Stage2 Group %d] Successfully synced weights\n", group_id);
                }
            } else {
                printf("[Stage2 Group %d] Failed to sync weights\n", group_id);
            }
            
            free(weights);
            loss_sum = 0.0;
        }
        
        // Progress logging
        if (processed_count % config->pipeline.sync_frequency == 0) {
            printf("[Stage2 Group %d] Processed %d images\n", group_id, processed_count);
            fflush(stdout);
        }
    }

    printf("[Stage2 Group %d] Completed processing %d images\n", group_id, processed_count);
    
    double end_time = time_in_socket_seconds();
    printf("[Stage2 Group %d] Training completed in %.2f seconds, total syncs: %d\n", 
           group_id, end_time - start_time, sync_count);
    fflush(stdout);
    
    // Save final model
    // THAY THẾ đoạn save cuối file worker_stage2.c:
    // Save final model  
    pipeline_stage_save(stage2, "stage2", group_id);
    printf("[Stage2 Group %d] Final model saved\n", group_id);

    // Save final model if configured
    // if (config->model.save_individual_stages) {
    //     char stage_path[512];
    //     snprintf(stage_path, sizeof(stage_path), "%s/stage2_group_%d", config->model.output_dir, group_id);
        
    //     // Create output directory
    //     char mkdir_cmd[512];
    //     snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p %s", config->model.output_dir);
    //     system(mkdir_cmd);
        
    //     pipeline_stage_save(stage2, "stage2", group_id);
    //     printf("[Stage2 Group %d] Final model saved to %s\n", group_id, stage_path);
    // }
    
    // Cleanup
    pipeline_stage_free(stage2);
    close(server_sock);
    free_config(config);
    
    return 0;
}