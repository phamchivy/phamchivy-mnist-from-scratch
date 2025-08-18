#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdarg.h> 
#include <unistd.h>  
#include "../neural/nn.h"
#include "../socket/socket_utils.h"

typedef enum {
    WEIGHT_TYPE_HIDDEN = 1,
    WEIGHT_TYPE_OUTPUT = 2
} WeightType;

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <port>\n", argv[0]);
        return 1;
    }
    
    int port = atoi(argv[1]);
    printf("[Parameter Server] Starting on port %d (Hybrid Mode)\n", port);
    fflush(stdout);
    
    srand(time(NULL));
    
    // Initialize separate elastic centers
    separate_elastic_center_init_hidden(300, 784);  // Hidden: 300x784
    separate_elastic_center_init_output(10, 300);   // Output: 10x300
    
    printf("[Parameter Server] Separate elastic centers initialized\n");
    fflush(stdout);
    
    // Setup server socket
    int server_sock = setup_server(port);
    printf("[Parameter Server] Server socket ready, waiting for workers...\n");
    fflush(stdout);
    
    int request_count = 0;
    
    while (1) {
        printf("[Parameter Server] Waiting for worker connection...\n");
        fflush(stdout);
        
        int client_sock = accept_client(server_sock);
        request_count++;
        
        double request_start = time_in_socket_seconds();
        
        printf("[Parameter Server] Worker connected (request #%d)\n", request_count);
        fflush(stdout);
        
        // Receive worker ID
        int worker_id;
        if (recv_all(client_sock, &worker_id, sizeof(int)) != sizeof(int)) {
            printf("[Parameter Server] Failed to receive worker ID\n");
            close(client_sock);
            continue;
        }
        
        // Receive weight type
        WeightType weight_type;
        if (recv_all(client_sock, &weight_type, sizeof(WeightType)) != sizeof(WeightType)) {
            printf("[Parameter Server] Failed to receive weight type\n");
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
        
        // Receive weights
        double comm_start_1 = time_in_socket_seconds();
        double* worker_weights = malloc(sizeof(double) * weight_count);
        if (recv_all(client_sock, worker_weights, sizeof(double) * weight_count) != sizeof(double) * weight_count) {
            printf("[Parameter Server] Failed to receive weights\n");
            free(worker_weights);
            close(client_sock);
            continue;
        }
        double comm_end_1 = time_in_socket_seconds();
        
        const char* weight_type_str = (weight_type == WEIGHT_TYPE_HIDDEN) ? "HIDDEN" : "OUTPUT";
        printf("[Parameter Server] Received %d %s weights from worker %d in %.3fms\n", 
               weight_count, weight_type_str, worker_id, (comm_end_1 - comm_start_1) * 1000);
        fflush(stdout);
        
        // Update appropriate elastic center
        double update_start = time_in_socket_seconds();
        if (weight_type == WEIGHT_TYPE_HIDDEN) {
            separate_elastic_center_update_hidden(worker_weights, weight_count, 0.001);
        } else {
            separate_elastic_center_update_output(worker_weights, weight_count, 0.001);
        }
        double update_end = time_in_socket_seconds();
        
        printf("[Parameter Server] Updated %s elastic center in %.3fms\n", 
               weight_type_str, (update_end - update_start) * 1000);
        fflush(stdout);
        
        // Get center weights
        int center_count;
        double* center_weights;
        if (weight_type == WEIGHT_TYPE_HIDDEN) {
            center_weights = separate_elastic_center_get_hidden_weights(&center_count);
        } else {
            center_weights = separate_elastic_center_get_output_weights(&center_count);
        }
        
        // Send center weights back
        double comm_start_2 = time_in_socket_seconds();
        if (send_all(client_sock, &center_count, sizeof(int)) == sizeof(int) &&
            send_all(client_sock, center_weights, sizeof(double) * center_count) == sizeof(double) * center_count) {
            double comm_end_2 = time_in_socket_seconds();
            printf("[Parameter Server] Sent %d %s center weights to worker %d in %.3fms\n", 
                   center_count, weight_type_str, worker_id, (comm_end_2 - comm_start_2) * 1000);
        } else {
            printf("[Parameter Server] Failed to send center to worker %d\n", worker_id);
        }
        fflush(stdout);
        
        double request_end = time_in_socket_seconds();
        double total_time = (request_end - request_start) * 1000;
        printf("[Parameter Server] Request #%d (%s) processed in %.3fms\n", 
               request_count, weight_type_str, total_time);
        fflush(stdout);
        
        // Cleanup
        free(worker_weights);
        free(center_weights);
        close(client_sock);
        
        if (request_count % 10 == 0) {
            printf("[Parameter Server] === Processed %d total requests ===\n", request_count);
            fflush(stdout);
        }
    }
    
    // Save final hybrid model after all training completed
    printf("[Parameter Server] Training completed, saving final model...\n");
    fflush(stdout);
    
    network_save_hybrid_final("server_logs");
    
    printf("[Parameter Server] Final hybrid model saved successfully!\n");
    fflush(stdout);
    
    // Cleanup
    separate_elastic_center_cleanup();
    return 0;
}