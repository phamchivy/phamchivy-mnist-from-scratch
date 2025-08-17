#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "../neural/nn.h"
#include "../util/img.h"
#include "../socket/socket_utils.h"
#include "../socket/pipeline_socket.h"

int main(int argc, char** argv) {
    if (argc < 5) {
        printf("Usage: %s <group_id> <listen_port> <server_ip> <server_port>\n", argv[0]);
        return 1;
    }
    
    int group_id = atoi(argv[1]);
    int listen_port = atoi(argv[2]);
    const char* server_ip = argv[3];
    int server_port = atoi(argv[4]);
    
    printf("[Stage2 Group %d] Starting, listening on port %d, server: %s:%d\n", 
           group_id, listen_port, server_ip, server_port);
    fflush(stdout);
    
    srand(time(NULL) + group_id + 100);
    
    // Initialize pipeline stage 2 (300 → 10)
    PipelineStage* stage2 = pipeline_stage_create(300, 10, 0.1);
    stage2->easgd_enabled = true;
    stage2->alpha = 0.01;
    stage2->beta = 0.001;
    
    printf("[Stage2 Group %d] Pipeline stage initialized: 300→10, α=0.01, β=0.001\n", group_id);
    fflush(stdout);
    
    // Setup server to receive activations from stage 1
    int server_sock = setup_activation_server(listen_port);
    if (server_sock < 0) {
        printf("[Stage2 Group %d] Failed to setup activation server\n", group_id);
        return 1;
    }
    
    printf("[Stage2 Group %d] Activation server ready on port %d\n", listen_port);
    fflush(stdout);
    
    int processed_count = 0;
    int sync_count = 0;
    double loss_sum = 0.0;
    double start_time = time_in_socket_seconds();
    
    while (1) {
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
        Matrix* hidden_activation = receive_activation(stage1_sock, 300, 1);
        if (hidden_activation == NULL) {
            close(stage1_sock);
            continue;
        }
        
        // Create target output
        Matrix* target = matrix_create(10, 1);
        target->entries[label][0] = 1.0;
        
        // Forward and backward pass through stage 2
        Matrix* grad_to_stage1;
        double loss = pipeline_stage2_backward(stage2, hidden_activation, target, &grad_to_stage1);
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
        if (processed_count % 1000 == 0) {
            printf("[Stage2 Group %d] Syncing after %d images (sync #%d), avg_loss=%.6f\n", 
                   group_id, processed_count, ++sync_count, loss_sum / 1000);
            
            // Get stage 2 weights
            int weight_count;
            double* weights = pipeline_stage2_get_weights(stage2, &weight_count);
            
            // Send to parameter server
            if (send_weights_to_parameter_server(server_ip, server_port, group_id, 
                                               WEIGHT_TYPE_OUTPUT, weights, weight_count) == 0) {
                printf("[Stage2 Group %d] Successfully synced weights\n", group_id);
            } else {
                printf("[Stage2 Group %d] Failed to sync weights\n", group_id);
            }
            
            free(weights);
            loss_sum = 0.0;
            
            // Check if we've processed enough (30k images per group)
            if (processed_count >= 30000) {
                printf("[Stage2 Group %d] Completed processing 30k images\n", group_id);
                break;
            }
        }
        
        if (processed_count % 1000 == 0) {
            printf("[Stage2 Group %d] Processed %d images\n", group_id, processed_count);
            fflush(stdout);
        }
    }
    
    double end_time = time_in_socket_seconds();
    printf("[Stage2 Group %d] Training completed in %.2f seconds, total syncs: %d\n", 
           group_id, end_time - start_time, sync_count);
    fflush(stdout);
    
    // Save final model
    // THAY THẾ đoạn save cuối file worker_stage2.c:
    // Save final model  
    pipeline_stage_save(stage2, "stage2", group_id);
    printf("[Stage2 Group %d] Final model saved\n", group_id);
    
    // Cleanup
    pipeline_stage_free(stage2);
    close(server_sock);
    
    return 0;
}