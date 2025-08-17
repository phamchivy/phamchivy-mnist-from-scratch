#include "pipeline_socket.h"
#include "socket_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <errno.h>

int setup_activation_server(int port) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket creation failed");
        return -1;
    }
    
    // Set socket options to reuse address
    int opt = 1;
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("setsockopt failed");
        close(server_fd);
        return -1;
    }
    
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;
    
    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind failed");
        close(server_fd);
        return -1;
    }
    
    if (listen(server_fd, 10) < 0) {
        perror("listen failed");
        close(server_fd);
        return -1;
    }
    
    return server_fd;
}

int connect_to_next_stage(const char* ip, int port) {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("socket creation failed");
        return -1;
    }
    
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    
    if (inet_pton(AF_INET, ip, &addr.sin_addr) <= 0) {
        perror("invalid address");
        close(sockfd);
        return -1;
    }
    
    if (connect(sockfd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("connection failed");
        close(sockfd);
        return -1;
    }
    
    return sockfd;
}

int send_activation(int sockfd, Matrix* activation) {
    if (activation == NULL) return -1;
    
    // Send matrix dimensions first
    int rows = activation->rows;
    int cols = activation->cols;
    
    if (send_all(sockfd, &rows, sizeof(int)) != sizeof(int) ||
        send_all(sockfd, &cols, sizeof(int)) != sizeof(int)) {
        return -1;
    }
    
    // Send matrix data
    for (int i = 0; i < rows; i++) {
        if (send_all(sockfd, activation->entries[i], sizeof(double) * cols) != sizeof(double) * cols) {
            return -1;
        }
    }
    
    return 0;
}

Matrix* receive_activation(int sockfd, int expected_rows, int expected_cols) {
    // Receive matrix dimensions
    int rows, cols;
    if (recv_all(sockfd, &rows, sizeof(int)) != sizeof(int) ||
        recv_all(sockfd, &cols, sizeof(int)) != sizeof(int)) {
        return NULL;
    }
    
    // Validate dimensions
    if (rows != expected_rows || cols != expected_cols) {
        printf("Dimension mismatch: expected %dx%d, got %dx%d\n", 
               expected_rows, expected_cols, rows, cols);
        return NULL;
    }
    
    // Create matrix and receive data
    Matrix* activation = matrix_create(rows, cols);
    if (activation == NULL) return NULL;
    
    for (int i = 0; i < rows; i++) {
        if (recv_all(sockfd, activation->entries[i], sizeof(double) * cols) != sizeof(double) * cols) {
            matrix_free(activation);
            return NULL;
        }
    }
    
    return activation;
}

int send_gradient(int sockfd, Matrix* gradient) {
    if (gradient == NULL) return -1;
    
    // Send matrix dimensions first
    int rows = gradient->rows;
    int cols = gradient->cols;
    
    if (send_all(sockfd, &rows, sizeof(int)) != sizeof(int) ||
        send_all(sockfd, &cols, sizeof(int)) != sizeof(int)) {
        return -1;
    }
    
    // Send matrix data
    for (int i = 0; i < rows; i++) {
        if (send_all(sockfd, gradient->entries[i], sizeof(double) * cols) != sizeof(double) * cols) {
            return -1;
        }
    }
    
    return 0;
}

Matrix* receive_gradient(int sockfd, int expected_rows, int expected_cols) {
    // Receive matrix dimensions
    int rows, cols;
    if (recv_all(sockfd, &rows, sizeof(int)) != sizeof(int) ||
        recv_all(sockfd, &cols, sizeof(int)) != sizeof(int)) {
        return NULL;
    }
    
    // Validate dimensions
    if (rows != expected_rows || cols != expected_cols) {
        printf("Gradient dimension mismatch: expected %dx%d, got %dx%d\n", 
               expected_rows, expected_cols, rows, cols);
        return NULL;
    }
    
    // Create matrix and receive data
    Matrix* gradient = matrix_create(rows, cols);
    if (gradient == NULL) return NULL;
    
    for (int i = 0; i < rows; i++) {
        if (recv_all(sockfd, gradient->entries[i], sizeof(double) * cols) != sizeof(double) * cols) {
            matrix_free(gradient);
            return NULL;
        }
    }
    
    return gradient;
}

int send_weights_to_parameter_server(const char* server_ip, int server_port,
                                   int worker_id, WeightType weight_type,
                                   double* weights, int weight_count) {
    double sync_start = time_in_socket_seconds();
    
    // Connect to parameter server
    int sockfd = connect_to_server(server_ip, server_port);
    if (sockfd < 0) {
        printf("[Worker %d] Failed to connect to parameter server\n", worker_id);
        return -1;
    }
    
    // Send worker ID
    if (send_all(sockfd, &worker_id, sizeof(int)) != sizeof(int)) {
        printf("[Worker %d] Failed to send worker ID\n", worker_id);
        close(sockfd);
        return -1;
    }
    
    // Send weight type
    if (send_all(sockfd, &weight_type, sizeof(WeightType)) != sizeof(WeightType)) {
        printf("[Worker %d] Failed to send weight type\n", worker_id);
        close(sockfd);
        return -1;
    }
    
    // Send weight count
    if (send_all(sockfd, &weight_count, sizeof(int)) != sizeof(int)) {
        printf("[Worker %d] Failed to send weight count\n", worker_id);
        close(sockfd);
        return -1;
    }
    
    // Send weights
    double comm_start = time_in_socket_seconds();
    if (send_all(sockfd, weights, sizeof(double) * weight_count) != sizeof(double) * weight_count) {
        printf("[Worker %d] Failed to send weights\n", worker_id);
        close(sockfd);
        return -1;
    }
    double comm_end = time_in_socket_seconds();
    
    const char* weight_type_str = (weight_type == WEIGHT_TYPE_HIDDEN) ? "HIDDEN" : "OUTPUT";
    printf("[Worker %d] Sent %d %s weights in %.3fms\n", 
           worker_id, weight_count, weight_type_str, (comm_end - comm_start) * 1000);
    
    // Receive center weights count
    int center_count;
    if (recv_all(sockfd, &center_count, sizeof(int)) != sizeof(int)) {
        printf("[Worker %d] Failed to receive center count\n", worker_id);
        close(sockfd);
        return -1;
    }
    
    // Receive center weights
    comm_start = time_in_socket_seconds();
    double* center_weights = malloc(sizeof(double) * center_count);
    if (recv_all(sockfd, center_weights, sizeof(double) * center_count) != sizeof(double) * center_count) {
        printf("[Worker %d] Failed to receive center weights\n", worker_id);
        free(center_weights);
        close(sockfd);
        return -1;
    }
    comm_end = time_in_socket_seconds();
    
    printf("[Worker %d] Received %d %s center weights in %.3fms\n", 
           worker_id, center_count, weight_type_str, (comm_end - comm_start) * 1000);
    
    // Apply elastic averaging based on weight type
    if (weight_type == WEIGHT_TYPE_HIDDEN) {
        // This will be handled by calling function
    } else {
        // This will be handled by calling function  
    }
    
    double sync_end = time_in_socket_seconds();
    printf("[Worker %d] %s sync completed in %.3fms\n", 
           worker_id, weight_type_str, (sync_end - sync_start) * 1000);
    
    // Note: center_weights should be freed by calling function after use
    free(center_weights);
    close(sockfd);
    return 0;
}

int receive_weights_from_parameter_server(int sockfd, WeightType expected_type,
                                        double** weights_out, int* count_out) {
    // Receive weight type
    WeightType weight_type;
    if (recv_all(sockfd, &weight_type, sizeof(WeightType)) != sizeof(WeightType)) {
        return -1;
    }
    
    if (weight_type != expected_type) {
        printf("Weight type mismatch: expected %d, got %d\n", expected_type, weight_type);
        return -1;
    }
    
    // Receive weight count
    int count;
    if (recv_all(sockfd, &count, sizeof(int)) != sizeof(int)) {
        return -1;
    }
    
    // Receive weights
    double* weights = malloc(sizeof(double) * count);
    if (recv_all(sockfd, weights, sizeof(double) * count) != sizeof(double) * count) {
        free(weights);
        return -1;
    }
    
    *weights_out = weights;
    *count_out = count;
    return 0;
}