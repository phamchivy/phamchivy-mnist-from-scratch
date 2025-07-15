/**
 * @file message.c
 * @brief Message handling and network communication for pipeline parallelism
 * @author Research Team
 * @date 2024
 * @version 1.0
 * 
 * This file implements message creation, serialization, and network communication
 * functions for the pipeline parallelism system. It handles forward and backward
 * message passing between pipeline stages with proper memory management and
 * error handling.
 * 
 * Key features:
 * - Message creation and destruction functions
 * - Network serialization and deserialization
 * - Timeout-based and non-blocking communication
 * - Statistics message handling
 * - Comprehensive error handling and recovery
 */

#include "pipeline_nn.h"
#include "../socket/socket_utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/select.h>

/* =============================================================================
 * MESSAGE CREATION FUNCTIONS
 * ============================================================================= */
ForwardMessage* create_forward_message(int batch_id, int mini_batch_id, Matrix* activations, int* labels, int label_count) {
    ForwardMessage* msg = (ForwardMessage*)malloc(sizeof(ForwardMessage));
    if (!msg) return NULL;
    
    msg->batch_id = batch_id;
    msg->mini_batch_id = mini_batch_id;
    msg->label_count = label_count;
    
    /* Copy activation data */
    msg->activation_count = activations->rows * activations->cols;
    msg->activations = (double*)malloc(sizeof(double) * msg->activation_count);
    if (!msg->activations) {
        free(msg);
        return NULL;
    }
    
    int idx = 0;
    for (int i = 0; i < activations->rows; i++) {
        for (int j = 0; j < activations->cols; j++) {
            msg->activations[idx++] = activations->entries[i][j];
        }
    }
    
    /* Copy labels */
    msg->labels = (int*)malloc(sizeof(int) * label_count);
    if (!msg->labels) {
        free(msg->activations);
        free(msg);
        return NULL;
    }
    memcpy(msg->labels, labels, sizeof(int) * label_count);
    
    return msg;
}

BackwardMessage* create_backward_message(int batch_id, int mini_batch_id, Matrix* gradients, double loss, int correct, int total) {
    BackwardMessage* msg = (BackwardMessage*)malloc(sizeof(BackwardMessage));
    if (!msg) return NULL;
    
    msg->batch_id = batch_id;
    msg->mini_batch_id = mini_batch_id;
    msg->loss = loss;
    msg->correct_predictions = correct;
    msg->total_predictions = total;
    
    /* Copy gradient data */
    msg->gradient_count = gradients->rows * gradients->cols;
    msg->gradients = (double*)malloc(sizeof(double) * msg->gradient_count);
    if (!msg->gradients) {
        free(msg);
        return NULL;
    }
    
    int idx = 0;
    for (int i = 0; i < gradients->rows; i++) {
        for (int j = 0; j < gradients->cols; j++) {
            msg->gradients[idx++] = gradients->entries[i][j];
        }
    }
    
    return msg;
}

void free_forward_message(ForwardMessage* msg) {
    if (!msg) return;
    if (msg->activations) free(msg->activations);
    if (msg->labels) free(msg->labels);
    free(msg);
}

void free_backward_message(BackwardMessage* msg) {
    if (!msg) return;
    if (msg->gradients) free(msg->gradients);
    free(msg);
}

/* =============================================================================
 * NETWORK COMMUNICATION FUNCTIONS
 * ============================================================================= */

/**
 * @brief Send forward activations message over network
 * @param sockfd Socket file descriptor
 * @param msg Pointer to forward message
 * @return 0 on success, -1 on failure
 */
int send_forward_activations(int sockfd, ForwardMessage* msg) {
    if (!msg) return -1;
    
    /* Send metadata first */
    if (send_all(sockfd, &msg->batch_id, sizeof(int)) != sizeof(int)) return -1;
    if (send_all(sockfd, &msg->mini_batch_id, sizeof(int)) != sizeof(int)) return -1;
    if (send_all(sockfd, &msg->activation_count, sizeof(int)) != sizeof(int)) return -1;
    if (send_all(sockfd, &msg->label_count, sizeof(int)) != sizeof(int)) return -1;
    
    /* Send activation data */
    int activation_bytes = sizeof(double) * msg->activation_count;
    if (send_all(sockfd, msg->activations, activation_bytes) != activation_bytes) return -1;
    
    /* Send labels */
    int label_bytes = sizeof(int) * msg->label_count;
    if (send_all(sockfd, msg->labels, label_bytes) != label_bytes) return -1;
    
    return 0;
}

ForwardMessage* receive_forward_activations(int sockfd) {
    ForwardMessage* msg = (ForwardMessage*)malloc(sizeof(ForwardMessage));
    if (!msg) return NULL;
    
    /* Receive metadata */
    if (recv_all(sockfd, &msg->batch_id, sizeof(int)) != sizeof(int)) goto error;
    if (recv_all(sockfd, &msg->mini_batch_id, sizeof(int)) != sizeof(int)) goto error;
    if (recv_all(sockfd, &msg->activation_count, sizeof(int)) != sizeof(int)) goto error;
    if (recv_all(sockfd, &msg->label_count, sizeof(int)) != sizeof(int)) goto error;
    
    /* Allocate and receive activations */
    msg->activations = (double*)malloc(sizeof(double) * msg->activation_count);
    if (!msg->activations) goto error;
    
    int activation_bytes = sizeof(double) * msg->activation_count;
    if (recv_all(sockfd, msg->activations, activation_bytes) != activation_bytes) goto error;
    
    /* Allocate and receive labels */
    msg->labels = (int*)malloc(sizeof(int) * msg->label_count);
    if (!msg->labels) goto error;
    
    int label_bytes = sizeof(int) * msg->label_count;
    if (recv_all(sockfd, msg->labels, label_bytes) != label_bytes) goto error;
    
    return msg;
    
error:
    free_forward_message(msg);
    return NULL;
}

int send_backward_gradients(int sockfd, BackwardMessage* msg) {
    if (!msg) return -1;
    
    /* Send metadata */
    if (send_all(sockfd, &msg->batch_id, sizeof(int)) != sizeof(int)) return -1;
    if (send_all(sockfd, &msg->mini_batch_id, sizeof(int)) != sizeof(int)) return -1;
    if (send_all(sockfd, &msg->gradient_count, sizeof(int)) != sizeof(int)) return -1;
    if (send_all(sockfd, &msg->loss, sizeof(double)) != sizeof(double)) return -1;
    if (send_all(sockfd, &msg->correct_predictions, sizeof(int)) != sizeof(int)) return -1;
    if (send_all(sockfd, &msg->total_predictions, sizeof(int)) != sizeof(int)) return -1;
    
    /* Send gradient data */
    int gradient_bytes = sizeof(double) * msg->gradient_count;
    if (send_all(sockfd, msg->gradients, gradient_bytes) != gradient_bytes) return -1;
    
    return 0;
}

BackwardMessage* receive_backward_gradients(int sockfd) {
    BackwardMessage* msg = (BackwardMessage*)malloc(sizeof(BackwardMessage));
    if (!msg) return NULL;
    
    /* Receive metadata */
    if (recv_all(sockfd, &msg->batch_id, sizeof(int)) != sizeof(int)) goto error;
    if (recv_all(sockfd, &msg->mini_batch_id, sizeof(int)) != sizeof(int)) goto error;
    if (recv_all(sockfd, &msg->gradient_count, sizeof(int)) != sizeof(int)) goto error;
    if (recv_all(sockfd, &msg->loss, sizeof(double)) != sizeof(double)) goto error;
    if (recv_all(sockfd, &msg->correct_predictions, sizeof(int)) != sizeof(int)) goto error;
    if (recv_all(sockfd, &msg->total_predictions, sizeof(int)) != sizeof(int)) goto error;
    
    /* Allocate and receive gradients */
    msg->gradients = (double*)malloc(sizeof(double) * msg->gradient_count);
    if (!msg->gradients) goto error;
    
    int gradient_bytes = sizeof(double) * msg->gradient_count;
    if (recv_all(sockfd, msg->gradients, gradient_bytes) != gradient_bytes) goto error;
    
    return msg;
    
error:
    free_backward_message(msg);
    return NULL;
}

/**
 * @brief Non-blocking version of receive_backward_gradients
 * @param sockfd Socket file descriptor
 * @return Pointer to backward message or NULL if no data available
 */
BackwardMessage* receive_backward_gradients_nonblocking(int sockfd) {
    /* Check if data is available without blocking */
    if (!has_pending_data(sockfd)) {
        return NULL; /* No data available, don't block */
    }
    
    /* Data is available, receive normally */
    return receive_backward_gradients(sockfd);
}

/**
 * @brief Blocking version with timeout of receive_backward_gradients
 * @param sockfd Socket file descriptor
 * @param timeout_ms Timeout in milliseconds
 * @return Pointer to backward message or NULL on timeout/error
 */
BackwardMessage* receive_backward_gradients_timeout(int sockfd, int timeout_ms) {
    fd_set readfds;
    struct timeval timeout;
    
    FD_ZERO(&readfds);
    FD_SET(sockfd, &readfds);
    
    timeout.tv_sec = timeout_ms / 1000;
    timeout.tv_usec = (timeout_ms % 1000) * 1000;
    
    int result = select(sockfd + 1, &readfds, NULL, NULL, &timeout);
    
    if (result > 0 && FD_ISSET(sockfd, &readfds)) {
        /* Data is available, receive it */
        return receive_backward_gradients(sockfd);
    }
    
    /* Timeout or no data available */
    return NULL;
}

int send_control_message(int sockfd, ControlMessage* msg) {
    if (!msg) return -1;
    
    if (send_all(sockfd, &msg->command, sizeof(ControlCommand)) != sizeof(ControlCommand)) return -1;
    if (send_all(sockfd, &msg->parameter, sizeof(double)) != sizeof(double)) return -1;
    
    return 0;
}

ControlMessage* receive_control_message(int sockfd) {
    ControlMessage* msg = (ControlMessage*)malloc(sizeof(ControlMessage));
    if (!msg) return NULL;
    
    if (recv_all(sockfd, &msg->command, sizeof(ControlCommand)) != sizeof(ControlCommand)) {
        free(msg);
        return NULL;
    }
    if (recv_all(sockfd, &msg->parameter, sizeof(double)) != sizeof(double)) {
        free(msg);
        return NULL;
    }
    
    return msg;
}

/* =============================================================================
 * STATISTICS COMMUNICATION FUNCTIONS
 * ============================================================================= */
int send_stats_message(int sockfd, StatsMessage* msg) {
    if (!msg) return -1;
    
    if (send_all(sockfd, &msg->stage_id, sizeof(int)) != sizeof(int)) return -1;
    if (send_all(sockfd, &msg->processing_time, sizeof(double)) != sizeof(double)) return -1;
    if (send_all(sockfd, &msg->communication_time, sizeof(double)) != sizeof(double)) return -1;
    if (send_all(sockfd, &msg->loss, sizeof(double)) != sizeof(double)) return -1;
    if (send_all(sockfd, &msg->batch_count, sizeof(int)) != sizeof(int)) return -1;
    if (send_all(sockfd, &msg->correct_predictions, sizeof(int)) != sizeof(int)) return -1;
    if (send_all(sockfd, &msg->total_predictions, sizeof(int)) != sizeof(int)) return -1;
    if (send_all(sockfd, &msg->timestamp, sizeof(long)) != sizeof(long)) return -1;
    
    return 0;
}

StatsMessage* receive_stats_message(int sockfd) {
    StatsMessage* msg = (StatsMessage*)malloc(sizeof(StatsMessage));
    if (!msg) return NULL;
    
    if (recv_all(sockfd, &msg->stage_id, sizeof(int)) != sizeof(int)) goto error;
    if (recv_all(sockfd, &msg->processing_time, sizeof(double)) != sizeof(double)) goto error;
    if (recv_all(sockfd, &msg->communication_time, sizeof(double)) != sizeof(double)) goto error;
    if (recv_all(sockfd, &msg->loss, sizeof(double)) != sizeof(double)) goto error;
    if (recv_all(sockfd, &msg->batch_count, sizeof(int)) != sizeof(int)) goto error;
    if (recv_all(sockfd, &msg->correct_predictions, sizeof(int)) != sizeof(int)) goto error;
    if (recv_all(sockfd, &msg->total_predictions, sizeof(int)) != sizeof(int)) goto error;
    if (recv_all(sockfd, &msg->timestamp, sizeof(long)) != sizeof(long)) goto error;
    
    return msg;
    
error:
    free(msg);
    return NULL;
}

int connect_stats_client(const char* coordinator_ip, int stats_port) {
    return connect_to_server(coordinator_ip, stats_port);
}