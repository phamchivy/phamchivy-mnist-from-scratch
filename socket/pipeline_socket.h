#ifndef PIPELINE_SOCKET_H
#define PIPELINE_SOCKET_H

#include "../matrix/matrix.h"

// Pipeline-specific socket functions
int setup_activation_server(int port);
int connect_to_next_stage(const char* ip, int port);
int send_activation(int sockfd, Matrix* activation);
Matrix* receive_activation(int sockfd, int expected_rows, int expected_cols);
int send_gradient(int sockfd, Matrix* gradient);
Matrix* receive_gradient(int sockfd, int expected_rows, int expected_cols);

// Weight communication với parameter server
typedef enum {
    WEIGHT_TYPE_HIDDEN = 1,
    WEIGHT_TYPE_OUTPUT = 2
} WeightType;

int send_weights_to_parameter_server(const char* server_ip, int server_port,
                                   int worker_id, WeightType weight_type,
                                   double* weights, int weight_count);

int receive_weights_from_parameter_server(int sockfd, WeightType expected_type,
                                        double** weights_out, int* count_out);

#endif