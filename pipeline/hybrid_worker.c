/**
 * @file hybrid_workers.c
 * @brief Hybrid Stage1 and Stage2 Worker Implementations
 * @author Research Team
 * @date 2024
 * 
 * This file implements the hybrid workers that combine pipeline parallelism
 * with EASGD synchronization. Stage1 workers process data partitions and
 * Stage2 worker handles multiplexed input from multiple Stage1 workers.
 */

#include "hybrid_pipeline_easgd.h"
#include "../socket/socket_utils.h"
#include <stdlib.h>      // malloc, free, getenv
#include <stdio.h>       // printf, fprintf, fopen, fclose, FILE, stdout, fflush
#include <string.h>      // memset, strcpy, strcmp
#include <netinet/in.h>  // struct sockaddr_in, socklen_t
#include <sys/socket.h>  // accept
#include <stdbool.h>
#include <unistd.h>

extern pthread_mutex_t stats_mutex;
void process_stage2_message(Stage2Worker* worker, ForwardMessage* fwd_msg, int source_worker);
void send_backward_response(Stage2Worker* worker, ForwardMessage* fwd_msg, int backward_socket);

/* Global training control flag */
static int global_training_active = 1;

/* =============================================================================
 * MESSAGE QUEUE IMPLEMENTATION
 * ============================================================================= */

int init_message_queue(MessageQueue* queue) {
    if (!queue) return -1;
    
    queue->head = 0;
    queue->tail = 0;
    queue->count = 0;
    
    if (pthread_mutex_init(&queue->mutex, NULL) != 0) {
        printf("[MSG_QUEUE] Error: Failed to initialize mutex\n");
        return -1;
    }
    
    if (pthread_cond_init(&queue->cond, NULL) != 0) {
        printf("[MSG_QUEUE] Error: Failed to initialize condition variable\n");
        pthread_mutex_destroy(&queue->mutex);
        return -1;
    }
    
    return 0;
}

int enqueue_message(MessageQueue* queue, ForwardMessage* msg) {
    if (!queue || !msg) return -1;
    
    pthread_mutex_lock(&queue->mutex);
    
    if (queue->count >= MAX_PIPELINE_DEPTH) {
        pthread_mutex_unlock(&queue->mutex);
        return -1; // Queue full
    }
    
    queue->messages[queue->tail] = msg;
    queue->tail = (queue->tail + 1) % MAX_PIPELINE_DEPTH;
    queue->count++;
    
    pthread_cond_signal(&queue->cond);
    pthread_mutex_unlock(&queue->mutex);
    
    return 0;
}

ForwardMessage* dequeue_message(MessageQueue* queue) {
    if (!queue) return NULL;
    
    pthread_mutex_lock(&queue->mutex);
    
    while (queue->count == 0 && global_training_active) {
        struct timespec timeout;
        clock_gettime(CLOCK_REALTIME, &timeout);
        timeout.tv_sec += 1; // 1 second timeout
        
        if (pthread_cond_timedwait(&queue->cond, &queue->mutex, &timeout) == ETIMEDOUT) {
            pthread_mutex_unlock(&queue->mutex);
            return NULL;
        }
    }
    
    if (queue->count == 0) {
        pthread_mutex_unlock(&queue->mutex);
        return NULL;
    }
    
    ForwardMessage* msg = queue->messages[queue->head];
    queue->head = (queue->head + 1) % MAX_PIPELINE_DEPTH;
    queue->count--;
    
    pthread_mutex_unlock(&queue->mutex);
    return msg;
}

void cleanup_message_queue(MessageQueue* queue) {
    if (!queue) return;
    
    pthread_mutex_lock(&queue->mutex);
    
    // Free any remaining messages
    while (queue->count > 0) {
        ForwardMessage* msg = queue->messages[queue->head];
        if (msg) free_forward_message(msg);
        queue->head = (queue->head + 1) % MAX_PIPELINE_DEPTH;
        queue->count--;
    }
    
    pthread_mutex_unlock(&queue->mutex);
    
    pthread_mutex_destroy(&queue->mutex);
    pthread_cond_destroy(&queue->cond);
}

/* =============================================================================
 * STAGE2 WORKER IMPLEMENTATION
 * ============================================================================= */

Stage2Worker* create_stage2_worker(double alpha2) {
    Stage2Worker* worker = (Stage2Worker*)malloc(sizeof(Stage2Worker));
    if (!worker) {
        printf("[STAGE2-1] Error: Failed to allocate worker structure\n");
        return NULL;
    }
    
    // Initialize worker parameters
    worker->worker_id = 1; // Single Stage2 worker
    worker->alpha2 = alpha2;
    worker->training_active = &global_training_active;
    worker->last_easgd_socket = -1; // Socket tạm thời cho EASGD mỗi epoch
    
    // Create Stage2 network (512 -> 10)
    worker->stage2 = create_stage(2, 512, 10, 0.01);
    if (!worker->stage2) {
        printf("[STAGE2-1] Error: Failed to create Stage2 network\n");
        free(worker);
        return NULL;
    }
    
    // Initialize message queues
    if (init_message_queue(&worker->stage1_1_queue) < 0 ||
        init_message_queue(&worker->stage1_2_queue) < 0) {
        printf("[STAGE2-1] Error: Failed to initialize message queues\n");
        free_stage(worker->stage2);
        free(worker);
        return NULL;
    }
    
    // Initialize sockets
    worker->stage1_1_forward_server = -1;
    worker->stage1_2_forward_server = -1;
    worker->stage1_1_backward_client = -1;
    worker->stage1_2_backward_client = -1;
    worker->stage1_1_forward_client = -1;
    worker->stage1_2_forward_client = -1;
    //worker->param_server_socket = -1;
    
    // Initialize EASGD components
    worker->stage2_weights_flat = NULL;
    worker->stage2_weight_count = 0;
    
    // Initialize metrics
    memset(&worker->local_stats, 0, sizeof(PipelineStats));
    worker->total_easgd_comm_time = 0.0;
    worker->messages_from_stage1_1 = 0;
    worker->messages_from_stage1_2 = 0;
    
    printf("[STAGE2-1] Created worker with α₂=%.3f\n", alpha2);
    return worker;
}

void hybrid_stage2_main(double alpha2) {
    printf("[STAGE2-1] Starting hybrid Stage2 worker with α₂=%.3f\n", alpha2);
    
    // Create Stage2 worker
    Stage2Worker* worker = create_stage2_worker(alpha2);
    if (!worker) {
        printf("[STAGE2-1] Failed to create worker\n");
        return;
    }
    
    // Setup pipeline servers for both Stage1 workers
    worker->stage1_1_forward_server = setup_server(8001);
    worker->stage1_2_forward_server = setup_server(8002);
    
    if (worker->stage1_1_forward_server < 0 || worker->stage1_2_forward_server < 0) {
        printf("[STAGE2-1] Failed to setup pipeline servers\n");
        cleanup_stage2_worker(worker);
        return;
    }
    
    printf("[STAGE2-1] Waiting for connections from Stage1 workers...\n");
    
    // Accept connections from Stage1 workers
    worker->stage1_1_forward_client = accept_client(worker->stage1_1_forward_server);
    worker->stage1_2_forward_client = accept_client(worker->stage1_2_forward_server);
    
    if (worker->stage1_1_forward_client < 0 || worker->stage1_2_forward_client < 0) {
        printf("[STAGE2-1] Failed to accept Stage1 connections\n");
        cleanup_stage2_worker(worker);
        return;
    }
    
    // Setup backward connections to Stage1 workers with retry mechanism
    printf("[STAGE2-1] Attempting to connect to Stage1 backward sockets...\n");
    
    worker->stage1_1_backward_client = -1;
    worker->stage1_2_backward_client = -1;
    
    // Retry connection to Stage1-1 backward server
    for (int retry = 0; retry < 30; retry++) {
        worker->stage1_1_backward_client = connect_to_server("172.33.0.2", 8003);
        if (worker->stage1_1_backward_client >= 0) {
            printf("[STAGE2-1] Connected to Stage1-1 backward server on attempt %d\n", retry + 1);
            break;
        }
        printf("[STAGE2-1] Stage1-1 backward connection attempt %d failed, retrying in 2s...\n", retry + 1);
        sleep(2);
    }
    
    // Retry connection to Stage1-2 backward server
    for (int retry = 0; retry < 30; retry++) {
        worker->stage1_2_backward_client = connect_to_server("172.33.0.3", 8004);
        if (worker->stage1_2_backward_client >= 0) {
            printf("[STAGE2-1] Connected to Stage1-2 backward server on attempt %d\n", retry + 1);
            break;
        }
        printf("[STAGE2-1] Stage1-2 backward connection attempt %d failed, retrying in 2s...\n", retry + 1);
        sleep(2);
    }
    
    if (worker->stage1_1_backward_client < 0 || worker->stage1_2_backward_client < 0) {
        printf("[STAGE2-1] Failed to connect to Stage1 backward sockets after multiple attempts\n");
        cleanup_stage2_worker(worker);
        return;
    }
    
    // Setup EASGD connection with retry mechanism
    printf("[STAGE2-1] Attempting to connect to parameter server...\n");
    
    // worker->param_server_socket = -1;
    // for (int retry = 0; retry < 30; retry++) {
    //     worker->param_server_socket = connect_to_server("172.33.0.5", EASGD_COMM_PORT);
    //     if (worker->param_server_socket >= 0) {
    //         printf("[STAGE2-1] Connected to parameter server on attempt %d\n", retry + 1);
    //         break;
    //     }
    //     printf("[STAGE2-1] Parameter server connection attempt %d failed, retrying in 2s...\n", retry + 1);
    //     sleep(2);
    // }
    
    // if (worker->param_server_socket < 0) {
    //     printf("[STAGE2-1] Failed to connect to parameter server after multiple attempts\n");
    //     cleanup_stage2_worker(worker);
    //     return;
    // }

    printf("[STAGE2-1] EASGD communication will be established per epoch\n");
    
    printf("[STAGE2-1] All connections established. Starting training...\n");
    
    // Training loop
    int epochs = 5;
    
    for (int epoch = 0; epoch < epochs; epoch++) {

        // === UPDATED EASGD SYNCHRONIZATION ===
        printf("[STAGE2-1] === EPOCH %d: EASGD Synchronization ===\n", epoch + 1);
        
        struct timeval easgd_start, easgd_end;
        gettimeofday(&easgd_start, NULL);
        
        // Connect to coordinator for this epoch
        printf("[STAGE2-1] Connecting to coordinator for EASGD sync...\n");
        worker->last_easgd_socket = connect_to_server("172.33.0.5", EASGD_COMM_PORT);
        
        if (worker->last_easgd_socket < 0) {
            printf("[STAGE2-1] Failed to connect to coordinator, retrying...\n");
            for (int retry = 1; retry <= 10; retry++) {
                sleep(2);
                worker->last_easgd_socket = connect_to_server("172.33.0.5", EASGD_COMM_PORT);
                if (worker->last_easgd_socket >= 0) {
                    printf("[STAGE2-1] Connected on retry %d\n", retry);
                    break;
                }
            }
        }
        
        if (worker->last_easgd_socket >= 0) {
            // Send weights to parameter server
            if (send_stage2_weights_to_server(worker, epoch) < 0) {
                printf("[STAGE2-1] Failed to send weights to server\n");
            } else {
                // Receive master weights using same socket
                if (receive_master_stage2_weights(worker) < 0) {
                    printf("[STAGE2-1] Failed to receive master weights\n");
                }
            }
            // Socket is closed in receive function
        } else {
            printf("[STAGE2-1] Skipping EASGD sync for epoch %d due to connection failure\n", 
                   epoch + 1);
        }
        
        gettimeofday(&easgd_end, NULL);
        worker->total_easgd_comm_time += get_time_diff(easgd_start, easgd_end);
        
        printf("[STAGE2-1] EASGD synchronization completed for epoch %d\n", epoch + 1);
    


        printf("[STAGE2-1] === EPOCH %d: Multiplexed Pipeline Processing ===\n", epoch + 1);
        
        // Reset epoch counters
        worker->messages_from_stage1_1 = 0;
        worker->messages_from_stage1_2 = 0;
        
        // Process multiplexed messages from both Stage1 workers
        while (global_training_active) {
            if (process_multiplexed_messages(worker) <= 0) {
                break; // No more messages for this epoch
            }
        }
        
        printf("[STAGE2-1] Epoch %d completed: processed %d messages from Stage1-1, %d from Stage1-2\n",
               epoch + 1, worker->messages_from_stage1_1, worker->messages_from_stage1_2);
        
        // EASGD synchronization phase
        printf("[STAGE2-1] === EPOCH %d: EASGD Synchronization ===\n", epoch + 1);
        
        gettimeofday(&easgd_start, NULL);
        
        // Send weights to parameter server
        if (send_stage2_weights_to_server(worker, epoch) < 0) {
            printf("[STAGE2-1] Failed to send weights to server\n");
            continue;
        }
        
        // Receive master weights
        if (receive_master_stage2_weights(worker) < 0) {
            printf("[STAGE2-1] Failed to receive master weights\n");
            continue;
        }
        
        gettimeofday(&easgd_end, NULL);
        worker->total_easgd_comm_time += get_time_diff(easgd_start, easgd_end);
        
        printf("[STAGE2-1] EASGD synchronization completed for epoch %d\n", epoch + 1);
    }
    
    printf("[STAGE2-1] Training completed\n");
    cleanup_stage2_worker(worker);
}

int process_multiplexed_messages(Stage2Worker* worker) {
    if (!worker) return -1;
    
    fd_set read_fds;
    struct timeval timeout = {0, 100000}; // 100ms timeout
    
    FD_ZERO(&read_fds);
    FD_SET(worker->stage1_1_forward_client, &read_fds);
    FD_SET(worker->stage1_2_forward_client, &read_fds);
    
    int max_fd = max(worker->stage1_1_forward_client, worker->stage1_2_forward_client);
    int activity = select(max_fd + 1, &read_fds, NULL, NULL, &timeout);
    
    if (activity < 0) {
        perror("select error in Stage2");
        return -1;
    }
    
    if (activity == 0) {
        return 0; // Timeout, no messages
    }
    
    int processed = 0;
    
    // Process message from Stage1-1
    if (FD_ISSET(worker->stage1_1_forward_client, &read_fds)) {
        ForwardMessage* fwd_msg = receive_forward_activations(worker->stage1_1_forward_client);
        if (fwd_msg) {
            process_stage2_message(worker, fwd_msg, 1);
            send_backward_response(worker, fwd_msg, worker->stage1_1_backward_client);
            free_forward_message(fwd_msg);
            worker->messages_from_stage1_1++;
            processed++;
        }
    }
    
    // Process message from Stage1-2
    if (FD_ISSET(worker->stage1_2_forward_client, &read_fds)) {
        ForwardMessage* fwd_msg = receive_forward_activations(worker->stage1_2_forward_client);
        if (fwd_msg) {
            process_stage2_message(worker, fwd_msg, 2);
            send_backward_response(worker, fwd_msg, worker->stage1_2_backward_client);
            free_forward_message(fwd_msg);
            worker->messages_from_stage1_2++;
            processed++;
        }
    }
    
    return processed;
}

void process_stage2_message(Stage2Worker* worker, ForwardMessage* fwd_msg, int source_worker) {
    if (!worker || !fwd_msg) return;
    
    struct timeval stage2_start, stage2_end;
    gettimeofday(&stage2_start, NULL);
    
    // Reconstruct activation matrix (512 x batch_size)
    Matrix* activations = matrix_create(512, fwd_msg->label_count);
    if (!activations) {
        printf("[STAGE2-1] Failed to create activation matrix\n");
        return;
    }
    
    int idx = 0;
    for (int i = 0; i < 512 && idx < fwd_msg->activation_count; i++) {
        for (int j = 0; j < fwd_msg->label_count && idx < fwd_msg->activation_count; j++) {
            activations->entries[i][j] = fwd_msg->activations[idx++];
        }
    }
    
    // Stage2 forward pass (512 -> 10)
    Matrix* predictions = stage2_forward(worker->stage2, activations);
    if (!predictions) {
        printf("[STAGE2-1] Stage2 forward pass failed\n");
        matrix_free(activations);
        return;
    }
    
    // Calculate loss and accuracy
    double loss = calculate_pipeline_loss(predictions, fwd_msg->labels, fwd_msg->label_count);
    int correct = calculate_pipeline_accuracy(predictions, fwd_msg->labels, fwd_msg->label_count);
    
    // Stage2 backward pass (only for training mode)
    Matrix* gradients = NULL;
    if (fwd_msg->batch_id != -1) { // Training mode
        gradients = stage2_backward(worker->stage2, predictions, fwd_msg->labels, 
                                   fwd_msg->label_count, activations, false);
    }
    
    gettimeofday(&stage2_end, NULL);
    double processing_time = get_time_diff(stage2_start, stage2_end);
    
    // Update local statistics
    pthread_mutex_lock(&stats_mutex);
    worker->local_stats.stage2_processing_time += processing_time;
    worker->local_stats.total_loss += loss;
    worker->local_stats.correct_predictions += correct;
    worker->local_stats.total_predictions += fwd_msg->label_count;
    worker->local_stats.processed_batches++;
    pthread_mutex_unlock(&stats_mutex);
    
    // Store gradients in forward message for backward response
    if (gradients) {
        // Convert gradients to flat array for backward message
        int grad_count = gradients->rows * gradients->cols;
        double* grad_array = (double*)malloc(sizeof(double) * grad_count);
        if (grad_array) {
            idx = 0;
            for (int i = 0; i < gradients->rows; i++) {
                for (int j = 0; j < gradients->cols; j++) {
                    grad_array[idx++] = gradients->entries[i][j];
                }
            }
            // Store in fwd_msg for use in backward response
            fwd_msg->activation_count = grad_count; // Reuse field for gradient count
            if (fwd_msg->activations) free(fwd_msg->activations);
            fwd_msg->activations = grad_array; // Reuse field for gradients
        }
        matrix_free(gradients);
    }
    
    // Store loss and accuracy for backward response
    fwd_msg->batch_id = (fwd_msg->batch_id == -1) ? -1 : fwd_msg->batch_id; // Preserve evaluation mode
    
    // Cleanup
    matrix_free(activations);
    matrix_free(predictions);
    
    if (fwd_msg->mini_batch_id % 100 == 0) {
        printf("[STAGE2-1] Processed batch %d from STAGE1-%d: loss=%.4f, acc=%.2f%%, time=%.3fs\n",
               fwd_msg->mini_batch_id, source_worker, loss, 
               (double)correct / fwd_msg->label_count * 100.0, processing_time);
    }
}

void send_backward_response(Stage2Worker* worker, ForwardMessage* fwd_msg, int backward_socket) {
    if (!worker || !fwd_msg) return;
    
    // Create backward message using the processed data stored in fwd_msg
    BackwardMessage* bwd_msg = (BackwardMessage*)malloc(sizeof(BackwardMessage));
    if (!bwd_msg) return;
    
    bwd_msg->batch_id = fwd_msg->batch_id;
    bwd_msg->mini_batch_id = fwd_msg->mini_batch_id;
    bwd_msg->gradient_count = fwd_msg->activation_count; // Reused field
    bwd_msg->gradients = fwd_msg->activations; // Reused field (transfer ownership)
    fwd_msg->activations = NULL; // Transfer ownership to backward message
    
    // Extract loss and accuracy from local stats (simplified approach)
    pthread_mutex_lock(&stats_mutex);
    bwd_msg->loss = worker->local_stats.total_loss / max(1, worker->local_stats.processed_batches);
    bwd_msg->correct_predictions = worker->local_stats.correct_predictions;
    bwd_msg->total_predictions = worker->local_stats.total_predictions;
    pthread_mutex_unlock(&stats_mutex);
    
    // Send backward message
    if (send_backward_gradients(backward_socket, bwd_msg) < 0) {
        printf("[STAGE2-1] Failed to send backward message\n");
    }
    
    free_backward_message(bwd_msg);
}

// === Similar fixes for Stage2 send/receive functions ===
int send_stage2_weights_to_server(Stage2Worker* worker, int epoch) {
    if (!worker || worker->last_easgd_socket < 0) {
        printf("[STAGE2-1] Invalid worker or socket\n");
        return -1;
    }
    
    // Flatten weights
    if (flatten_stage2_weights(worker) < 0) {
        printf("[STAGE2-1] Failed to flatten weights\n");
        return -1;
    }
    
    // Create EASGD message
    EASGDMessage* msg = create_easgd_message(EASGD_WEIGHT_UPDATE, worker->worker_id, 2);
    if (!msg) return -1;
    
    msg->epoch = epoch;
    msg->stage2_weight_count = worker->stage2_weight_count;
    msg->stage2_weights = (double*)malloc(sizeof(double) * worker->stage2_weight_count);
    
    if (!msg->stage2_weights) {
        free_easgd_message(msg);
        return -1;
    }
    
    memcpy(msg->stage2_weights, worker->stage2_weights_flat, 
           sizeof(double) * worker->stage2_weight_count);
    
    // Add metadata
    msg->local_loss = worker->local_stats.total_loss / max(1, worker->local_stats.processed_batches);
    msg->local_accuracy = (double)worker->local_stats.correct_predictions / 
                         max(1, worker->local_stats.total_predictions) * 100.0;
    msg->processed_samples = worker->local_stats.total_predictions;
    
    printf("[STAGE2-1] Sending %d weights to coordinator...\n", worker->stage2_weight_count);
    
    if (send_easgd_message(worker->last_easgd_socket, msg) < 0) {
        printf("[STAGE2-1] Failed to send weights to parameter server\n");
        free_easgd_message(msg);
        return -1;
    }
    
    printf("[STAGE2-1] Weights sent successfully\n");
    free_easgd_message(msg);
    return 0;
}

int receive_master_stage2_weights(Stage2Worker* worker) {
    if (!worker || worker->last_easgd_socket < 0) {
        printf("[STAGE2-1] Invalid worker or socket\n");
        return -1;
    }
    
    printf("[STAGE2-1] Waiting for master weights from coordinator...\n");
    
    EASGDMessage* msg = receive_easgd_message(worker->last_easgd_socket);
    
    if (!msg || msg->message_type != EASGD_MASTER_WEIGHTS) {
        printf("[STAGE2-1] Failed to receive master weights\n");
        if (msg) free_easgd_message(msg);
        socket_close(worker->last_easgd_socket);
        worker->last_easgd_socket = -1;
        return -1;
    }
    
    printf("[STAGE2-1] Received %d master weights\n", msg->stage2_weight_count);
    
    if (apply_elastic_averaging_stage2(worker, msg->stage2_weights) < 0) {
        printf("[STAGE2-1] Failed to apply elastic averaging\n");
        free_easgd_message(msg);
        socket_close(worker->last_easgd_socket);
        worker->last_easgd_socket = -1;
        return -1;
    }
    
    printf("[STAGE2-1] Applied elastic averaging successfully\n");
    free_easgd_message(msg);
    socket_close(worker->last_easgd_socket);
    worker->last_easgd_socket = -1;
    return 0;
}

void cleanup_stage2_worker(Stage2Worker* worker) {
    if (!worker) return;
    
    // Cleanup network
    if (worker->stage2) free_stage(worker->stage2);
    if (worker->stage2_weights_flat) free(worker->stage2_weights_flat);
    
    // Cleanup sockets
    if (worker->stage1_1_forward_server >= 0) socket_close(worker->stage1_1_forward_server);
    if (worker->stage1_2_forward_server >= 0) socket_close(worker->stage1_2_forward_server);
    if (worker->stage1_1_forward_client >= 0) socket_close(worker->stage1_1_forward_client);
    if (worker->stage1_2_forward_client >= 0) socket_close(worker->stage1_2_forward_client);
    if (worker->stage1_1_backward_client >= 0) socket_close(worker->stage1_1_backward_client);
    if (worker->stage1_2_backward_client >= 0) socket_close(worker->stage1_2_backward_client);
    //if (worker->param_server_socket >= 0) socket_close(worker->param_server_socket);
    if (worker->last_easgd_socket >= 0) socket_close(worker->last_easgd_socket);
    
    // Cleanup message queues
    cleanup_message_queue(&worker->stage1_1_queue);
    cleanup_message_queue(&worker->stage1_2_queue);
    
    free(worker);
}

/* =============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================= */

BatchContext* find_pending_batch(PipelineBatchTracker* tracker, int batch_id) {
    if (!tracker) return NULL;
    
    pthread_mutex_lock(&tracker->tracker_mutex);
    
    for (int i = 0; i < tracker->pending_count; i++) {
        if (tracker->pending_batches[i].batch_id == batch_id) {
            pthread_mutex_unlock(&tracker->tracker_mutex);
            return &tracker->pending_batches[i];
        }
    }
    
    pthread_mutex_unlock(&tracker->tracker_mutex);
    return NULL;
}

int remove_pending_batch(PipelineBatchTracker* tracker, int batch_id) {
    if (!tracker) return -1;
    
    pthread_mutex_lock(&tracker->tracker_mutex);
    
    for (int i = 0; i < tracker->pending_count; i++) {
        if (tracker->pending_batches[i].batch_id == batch_id) {
            // Free saved matrices
            if (tracker->pending_batches[i].saved_activations) {
                matrix_free(tracker->pending_batches[i].saved_activations);
            }
            if (tracker->pending_batches[i].saved_inputs) {
                matrix_free(tracker->pending_batches[i].saved_inputs);
            }
            
            // Compact array by moving last element to this position
            if (i < tracker->pending_count - 1) {
                tracker->pending_batches[i] = tracker->pending_batches[tracker->pending_count - 1];
            }
            
            tracker->pending_count--;
            pthread_mutex_unlock(&tracker->tracker_mutex);
            return 0;
        }
    }
    
    pthread_mutex_unlock(&tracker->tracker_mutex);
    return -1; // Batch not found
}

/* =============================================================================
 * LOAD BALANCING AND OPTIMIZATION
 * ============================================================================= */

void balance_stage2_load(Stage2Worker* worker) {
    if (!worker) return;
    
    // Monitor message rates from both Stage1 workers
    static int prev_stage1_1_count = 0;
    static int prev_stage1_2_count = 0;
    static time_t last_balance_time = 0;
    
    time_t current_time = time(NULL);
    if (current_time - last_balance_time < 5) {
        return; // Only balance every 5 seconds
    }
    
    int rate1 = worker->messages_from_stage1_1 - prev_stage1_1_count;
    int rate2 = worker->messages_from_stage1_2 - prev_stage1_2_count;
    
    if (abs(rate1 - rate2) > 10) {
        printf("[STAGE2-1] Load imbalance detected: Stage1-1=%d msgs/5s, Stage1-2=%d msgs/5s\n", 
               rate1, rate2);
        
        // Could implement adaptive processing priority here
        // For now, just log the imbalance
    }
    
    prev_stage1_1_count = worker->messages_from_stage1_1;
    prev_stage1_2_count = worker->messages_from_stage1_2;
    last_balance_time = current_time;
}

void log_hybrid_worker_performance(Stage1Worker* stage1_worker, Stage2Worker* stage2_worker) {
    if (stage1_worker) {
        printf("HYBRID_LOG|%ld|STAGE1-%d|processing_time_%.3f|comm_time_%.3f|easgd_syncs_%d\n",
               time(NULL), stage1_worker->worker_id,
               stage1_worker->local_stats.stage1_processing_time,
               stage1_worker->local_stats.communication_time,
               stage1_worker->easgd_sync_count);
    }
    
    if (stage2_worker) {
        printf("HYBRID_LOG|%ld|STAGE2-1|processing_time_%.3f|msg1_%d|msg2_%d|loss_%.4f|acc_%.2f\n",
               time(NULL),
               stage2_worker->local_stats.stage2_processing_time,
               stage2_worker->messages_from_stage1_1,
               stage2_worker->messages_from_stage1_2,
               stage2_worker->local_stats.total_loss / max(1, stage2_worker->local_stats.processed_batches),
               (double)stage2_worker->local_stats.correct_predictions / 
               max(1, stage2_worker->local_stats.total_predictions) * 100.0);
    }
    
    fflush(stdout);
}/**
 * @file hybrid_workers.c
 * @brief Hybrid Stage1 and Stage2 Worker Implementations
 * @author Research Team
 * @date 2024
 * 
 * This file implements the hybrid workers that combine pipeline parallelism
 * with EASGD synchronization. Stage1 workers process data partitions and
 * Stage2 worker handles multiplexed input from multiple Stage1 workers.
 */

/* =============================================================================
 * STAGE1 WORKER IMPLEMENTATION
 * ============================================================================= */

Stage1Worker* create_stage1_worker(int worker_id, int data_start, int data_end, double alpha1) {
    Stage1Worker* worker = (Stage1Worker*)malloc(sizeof(Stage1Worker));
    if (!worker) {
        printf("[STAGE1-%d] Error: Failed to allocate worker structure\n", worker_id);
        return NULL;
    }
    
    // Initialize worker parameters
    worker->worker_id = worker_id;
    worker->data_start_idx = data_start;
    worker->data_end_idx = data_end;
    worker->alpha1 = alpha1;
    worker->training_active = &global_training_active;
    worker->last_easgd_socket = -1; // Socket tạm thời cho EASGD mỗi epoch
    
    // Create Stage1 network (784 -> 512)
    worker->stage1 = create_stage(1, 784, 512, 0.01);
    if (!worker->stage1) {
        printf("[STAGE1-%d] Error: Failed to create Stage1 network\n", worker_id);
        free(worker);
        return NULL;
    }
    
    // Initialize pipeline components
    if (initialize_buffer(&worker->pipeline_buffer) < 0) {
        printf("[STAGE1-%d] Error: Failed to initialize pipeline buffer\n", worker_id);
        free_stage(worker->stage1);
        free(worker);
        return NULL;
    }
    
    if (initialize_batch_tracker(&worker->batch_tracker) < 0) {
        printf("[STAGE1-%d] Error: Failed to initialize batch tracker\n", worker_id);
        cleanup_buffer(&worker->pipeline_buffer);
        free_stage(worker->stage1);
        free(worker);
        return NULL;
    }
    
    // Initialize adaptive configuration
    worker->config.current_batch_size = MINI_BATCH_SIZE;
    worker->config.pipeline_depth = 4;
    worker->config.target_latency = 0.1;
    worker->config.communication_ratio = 0.0;
    worker->config.bubble_rate = 0.0;
    worker->config.last_adjustment_time = 0;
    
    // Initialize EASGD components
    worker->stage1_weights_flat = NULL;
    worker->stage1_weight_count = 0;
    //worker->param_server_socket = -1;
    worker->stage2_forward_socket = -1;
    worker->stage2_backward_socket = -1;
    worker->stage2_backward_server = -1;
    
    // Initialize metrics
    memset(&worker->local_stats, 0, sizeof(PipelineStats));
    worker->total_easgd_comm_time = 0.0;
    worker->easgd_sync_count = 0;
    
    printf("[STAGE1-%d] Created worker for data partition [%d:%d] with α₁=%.3f\n",
           worker_id, data_start, data_end, alpha1);
    
    return worker;
}

void* hybrid_stage1_forward_processor(void* args) {
    AsyncForwardArgs* fwd_args = (AsyncForwardArgs*)args;
    Stage1Worker* worker = (Stage1Worker*)fwd_args->stage;
    
    printf("[STAGE1-%d] Forward processor thread started\n", worker->worker_id);
    
    while (*worker->training_active) {
        // Dequeue message from pipeline buffer
        ForwardMessage* fwd_msg = dequeue_forward(&worker->pipeline_buffer);
        if (!fwd_msg) {
            usleep(1000); // 1ms wait if no message
            continue;
        }
        
        // Send to Stage2-1
        struct timeval send_start, send_end;
        gettimeofday(&send_start, NULL);
        
        if (send_forward_activations(worker->stage2_forward_socket, fwd_msg) < 0) {
            printf("[STAGE1-%d] Failed to send message to Stage2-1\n", worker->worker_id);
            track_pipeline_bubble("Stage1 to Stage2 communication failure");
        } else {
            // Update communication statistics
            gettimeofday(&send_end, NULL);
            double comm_time = get_time_diff(send_start, send_end);
            
            pthread_mutex_lock(&stats_mutex);
            worker->local_stats.communication_time += comm_time;
            pthread_mutex_unlock(&stats_mutex);
        }
        
        free_forward_message(fwd_msg);
    }
    
    printf("[STAGE1-%d] Forward processor thread terminated\n", worker->worker_id);
    return NULL;
}

void* hybrid_stage1_backward_processor(void* args) {
    AsyncBackwardArgs* bwd_args = (AsyncBackwardArgs*)args;
    Stage1Worker* worker = (Stage1Worker*)bwd_args->stage;
    
    printf("[STAGE1-%d] Backward processor thread started\n", worker->worker_id);
    
    while (*worker->training_active) {
        // Receive gradients from Stage2-1
        BackwardMessage* bwd_msg = receive_backward_gradients(worker->stage2_backward_socket);
        if (!bwd_msg) {
            usleep(1000);
            continue;
        }
        
        // Find corresponding batch in tracker
        BatchContext* pending = find_pending_batch(&worker->batch_tracker, bwd_msg->batch_id);
        if (!pending) {
            printf("[STAGE1-%d] Warning: Received gradients for unknown batch %d\n", 
                   worker->worker_id, bwd_msg->batch_id);
            free_backward_message(bwd_msg);
            continue;
        }
        
        // Apply gradients to Stage1 weights (skip for evaluation mode)
        if (bwd_msg->batch_id != -1) {
            // Reconstruct Stage1 gradients and apply them
            Matrix* stage1_gradients = matrix_create(worker->stage1->weights->rows, 
                                                   worker->stage1->weights->cols);
            
            // Copy gradients from message (simplified - in practice need proper reconstruction)
            int grad_idx = 0;
            for (int i = 0; i < stage1_gradients->rows && grad_idx < bwd_msg->gradient_count; i++) {
                for (int j = 0; j < stage1_gradients->cols && grad_idx < bwd_msg->gradient_count; j++) {
                    stage1_gradients->entries[i][j] = bwd_msg->gradients[grad_idx++];
                }
            }
            
            // Apply gradients using learning rate
            for (int i = 0; i < worker->stage1->weights->rows; i++) {
                for (int j = 0; j < worker->stage1->weights->cols; j++) {
                    worker->stage1->weights->entries[i][j] -= 
                        worker->stage1->learning_rate * stage1_gradients->entries[i][j];
                }
            }
            
            matrix_free(stage1_gradients);
        }
        
        // Remove from batch tracker
        remove_pending_batch(&worker->batch_tracker, bwd_msg->batch_id);
        
        // Update local statistics
        pthread_mutex_lock(&stats_mutex);
        worker->local_stats.total_loss += bwd_msg->loss;
        worker->local_stats.correct_predictions += bwd_msg->correct_predictions;
        worker->local_stats.total_predictions += bwd_msg->total_predictions;
        pthread_mutex_unlock(&stats_mutex);
        
        free_backward_message(bwd_msg);
    }
    
    printf("[STAGE1-%d] Backward processor thread terminated\n", worker->worker_id);
    return NULL;
}

void hybrid_stage1_main(int worker_id, int data_start, int data_end, double alpha1) {
    printf("[STAGE1-%d] Starting hybrid worker for data partition [%d:%d]\n", 
           worker_id, data_start, data_end);
    
    // Create Stage1 worker
    Stage1Worker* worker = create_stage1_worker(worker_id, data_start, data_end, alpha1);
    if (!worker) {
        printf("[STAGE1-%d] Failed to create worker\n", worker_id);
        return;
    }
    
    // Load data partition
    char dataset_path[] = "./data/mnist_train.csv";
    Img** worker_data = load_dataset_partition(data_start, data_end, dataset_path);
    if (!worker_data) {
        printf("[STAGE1-%d] Failed to load data partition\n", worker_id);
        cleanup_stage1_worker(worker);
        return;
    }
    
    int worker_data_size = data_end - data_start;
    printf("[STAGE1-%d] Loaded %d training images\n", worker_id, worker_data_size);
    
    // Setup pipeline connections with retry mechanism
    printf("[STAGE1-%d] Attempting to connect to Stage2-1...\n", worker_id);
    
    worker->stage2_forward_socket = -1;
    worker->stage2_backward_socket = -1;
    
    // Retry connection to Stage2 forward server
    for (int retry = 0; retry < 30; retry++) {
        worker->stage2_forward_socket = connect_to_server("172.33.0.4", 8001 + worker_id - 1);
        if (worker->stage2_forward_socket >= 0) {
            printf("[STAGE1-%d] Connected to Stage2 forward server on attempt %d\n", worker_id, retry + 1);
            break;
        }
        printf("[STAGE1-%d] Forward connection attempt %d failed, retrying in 2s...\n", worker_id, retry + 1);
        sleep(2);
    }
    
    if (worker->stage2_forward_socket < 0) {
        printf("[STAGE1-%d] Failed to connect to Stage2-1 after multiple attempts\n", worker_id);
        cleanup_stage1_worker(worker);
        return;
    }
    
    // Setup backward server for Stage2 to connect
    printf("[STAGE1-%d] Setting up backward server on port %d...\n", worker_id, 8003 + worker_id - 1);
    worker->stage2_backward_server = setup_server(8003 + worker_id - 1);
    if (worker->stage2_backward_server < 0) {
        printf("[STAGE1-%d] Failed to setup backward server\n", worker_id);
        cleanup_stage1_worker(worker);
        return;
    }
    
    // Accept backward connection from Stage2
    printf("[STAGE1-%d] Waiting for backward connection from Stage2...\n", worker_id);
    worker->stage2_backward_socket = accept_client(worker->stage2_backward_server);
    if (worker->stage2_backward_socket < 0) {
        printf("[STAGE1-%d] Failed to accept backward connection from Stage2\n", worker_id);
        cleanup_stage1_worker(worker);
        return;
    }
    printf("[STAGE1-%d] Accepted backward connection from Stage2\n", worker_id);
    
    // Setup EASGD connection with retry mechanism
    printf("[STAGE1-%d] Attempting to connect to parameter server...\n", worker_id);
    
    // worker->param_server_socket = -1;
    // for (int retry = 0; retry < 30; retry++) {
    //     worker->param_server_socket = connect_to_server("172.33.0.5", EASGD_COMM_PORT);
    //     if (worker->param_server_socket >= 0) {
    //         printf("[STAGE1-%d] Connected to parameter server on attempt %d\n", worker_id, retry + 1);
    //         break;
    //     }
    //     printf("[STAGE1-%d] Parameter server connection attempt %d failed, retrying in 2s...\n", worker_id, retry + 1);
    //     sleep(2);
    // }
    
    // if (worker->param_server_socket < 0) {
    //     printf("[STAGE1-%d] Failed to connect to parameter server after multiple attempts\n", worker_id);
    //     cleanup_stage1_worker(worker);
    //     return;
    // }

    printf("[STAGE1-%d] EASGD communication will be established per epoch\n", worker_id);

    
    
    // Start async processing threads
    AsyncForwardArgs forward_args = {
        .stage = (NetworkStage*)worker,
        .buffer = &worker->pipeline_buffer,
        .tracker = &worker->batch_tracker,
        .forward_socket = worker->stage2_forward_socket,
        .training_active = worker->training_active,
        .config = &worker->config
    };
    
    AsyncBackwardArgs backward_args = {
        .stage = (NetworkStage*)worker,
        .buffer = &worker->pipeline_buffer,
        .tracker = &worker->batch_tracker,
        .backward_socket = worker->stage2_backward_socket,
        .training_active = worker->training_active,
        .config = &worker->config
    };
    
    if (pthread_create(&worker->forward_thread, NULL, hybrid_stage1_forward_processor, &forward_args) != 0 ||
        pthread_create(&worker->backward_thread, NULL, hybrid_stage1_backward_processor, &backward_args) != 0) {
        printf("[STAGE1-%d] Failed to create async threads\n", worker_id);
        cleanup_stage1_worker(worker);
        return;
    }
    
    // Training loop
    int epochs = 5;
    int current_batch_id = worker_id * 100000; // Unique batch IDs per worker
    
    for (int epoch = 0; epoch < epochs; epoch++) {

        // === UPDATED EASGD SYNCHRONIZATION ===
        printf("[STAGE1-%d] === EPOCH %d: EASGD Synchronization ===\n", worker_id, epoch + 1);
        
        struct timeval easgd_start, easgd_end;
        gettimeofday(&easgd_start, NULL);
        
        // Connect to coordinator for this epoch
        printf("[STAGE1-%d] Connecting to coordinator for EASGD sync...\n", worker_id);
        worker->last_easgd_socket = connect_to_server("172.33.0.5", EASGD_COMM_PORT);
        
        if (worker->last_easgd_socket < 0) {
            printf("[STAGE1-%d] Failed to connect to coordinator, retrying...\n", worker_id);
            for (int retry = 1; retry <= 10; retry++) {
                sleep(2);
                worker->last_easgd_socket = connect_to_server("172.33.0.5", EASGD_COMM_PORT);
                if (worker->last_easgd_socket >= 0) {
                    printf("[STAGE1-%d] Connected on retry %d\n", worker_id, retry);
                    break;
                }
            }
        }
        
        if (worker->last_easgd_socket >= 0) {
            // Send weights to parameter server
            if (send_stage1_weights_to_server(worker, epoch) < 0) {
                printf("[STAGE1-%d] Failed to send weights to server\n", worker_id);
            } else {
                // Receive master weights using same socket
                if (receive_master_stage1_weights(worker) < 0) {
                    printf("[STAGE1-%d] Failed to receive master weights\n", worker_id);
                }
            }
            // Socket is closed in receive function
        } else {
            printf("[STAGE1-%d] Skipping EASGD sync for epoch %d due to connection failure\n", 
                   worker_id, epoch + 1);
        }
        
        gettimeofday(&easgd_end, NULL);
        worker->total_easgd_comm_time += get_time_diff(easgd_start, easgd_end);
        worker->easgd_sync_count++;
        
        printf("[STAGE1-%d] EASGD synchronization completed for epoch %d\n", worker_id, epoch + 1);

        printf("[STAGE1-%d] === EPOCH %d: Pipeline Training Phase ===\n", worker_id, epoch + 1);
        
        // Pipeline training phase
        for (int i = 0; i < worker_data_size; i += worker->config.current_batch_size) {
            if (!global_training_active) break;
            
            int batch_size = min(worker->config.current_batch_size, worker_data_size - i);
            Img** mini_batch = &worker_data[i];
            
            // Extract labels
            int* labels = (int*)malloc(sizeof(int) * batch_size);
            for (int j = 0; j < batch_size; j++) {
                labels[j] = mini_batch[j]->label;
            }
            
            // Stage1 forward pass
            Matrix* input_matrix = matrix_create(784, batch_size);
            for (int img_idx = 0; img_idx < batch_size; img_idx++) {
                Img* img = mini_batch[img_idx];
                for (int pixel_row = 0; pixel_row < 28; pixel_row++) {
                    for (int pixel_col = 0; pixel_col < 28; pixel_col++) {
                        int pixel_idx = pixel_row * 28 + pixel_col;
                        input_matrix->entries[pixel_idx][img_idx] = 
                            img->img_data->entries[pixel_row][pixel_col];
                    }
                }
            }
            
            Matrix* hidden_activations = stage1_forward(worker->stage1, mini_batch, batch_size);
            if (!hidden_activations) {
                printf("[STAGE1-%d] Forward pass failed\n", worker_id);
                matrix_free(input_matrix);
                free(labels);
                continue;
            }
            
            // Add to batch tracker
            Matrix* saved_activations = matrix_create(hidden_activations->rows, hidden_activations->cols);
            Matrix* saved_inputs = matrix_create(input_matrix->rows, input_matrix->cols);
            
            for (int r = 0; r < hidden_activations->rows; r++) {
                for (int c = 0; c < hidden_activations->cols; c++) {
                    saved_activations->entries[r][c] = hidden_activations->entries[r][c];
                }
            }
            for (int r = 0; r < input_matrix->rows; r++) {
                for (int c = 0; c < input_matrix->cols; c++) {
                    saved_inputs->entries[r][c] = input_matrix->entries[r][c];
                }
            }
            
            add_pending_batch(&worker->batch_tracker, current_batch_id, saved_activations, saved_inputs);
            
            // Create and enqueue forward message
            ForwardMessage* fwd_msg = create_forward_message(
                current_batch_id, i / batch_size, hidden_activations, labels, batch_size
            );
            
            if (enqueue_forward(&worker->pipeline_buffer, fwd_msg) < 0) {
                printf("[STAGE1-%d] Pipeline buffer full, waiting...\n", worker_id);
                usleep(10000); // 10ms wait
                enqueue_forward(&worker->pipeline_buffer, fwd_msg);
            }
            
            current_batch_id++;
            
            // Adaptive optimization
            if ((i / batch_size) % 10 == 0) {
                PipelineStats current_stats = collect_pipeline_stats();
                update_pipeline_config(&worker->config, &current_stats);
            }
            
            // Cleanup
            matrix_free(hidden_activations);
            matrix_free(input_matrix);
            free(labels);
        }
        
        // Wait for pipeline to complete epoch
        printf("[STAGE1-%d] Waiting for pipeline completion...\n", worker_id);
        while (worker->batch_tracker.pending_count > 0 && global_training_active) {
            usleep(100000); // 100ms wait
        }
        
        // EASGD synchronization phase
        printf("[STAGE1-%d] === EPOCH %d: EASGD Synchronization ===\n", worker_id, epoch + 1);
        
        gettimeofday(&easgd_start, NULL);
        
        // Send weights to parameter server
        if (send_stage1_weights_to_server(worker, epoch) < 0) {
            printf("[STAGE1-%d] Failed to send weights to server\n", worker_id);
            continue;
        }
        
        // Receive master weights
        if (receive_master_stage1_weights(worker) < 0) {
            printf("[STAGE1-%d] Failed to receive master weights\n", worker_id);
            continue;
        }
        
        gettimeofday(&easgd_end, NULL);
        worker->total_easgd_comm_time += get_time_diff(easgd_start, easgd_end);
        worker->easgd_sync_count++;
        
        printf("[STAGE1-%d] EASGD synchronization completed for epoch %d\n", worker_id, epoch + 1);
    }
    
    // Signal threads to stop
    global_training_active = 0;
    pthread_cond_broadcast(&worker->pipeline_buffer.forward_cond);
    pthread_cond_broadcast(&worker->pipeline_buffer.backward_cond);
    
    // Wait for threads to complete
    pthread_join(worker->forward_thread, NULL);
    pthread_join(worker->backward_thread, NULL);
    
    printf("[STAGE1-%d] Training completed\n", worker_id);
    
    // Cleanup
    imgs_free(worker_data, worker_data_size);
    cleanup_stage1_worker(worker);
}

// === FIX send_stage1_weights_to_server function ===
int send_stage1_weights_to_server(Stage1Worker* worker, int epoch) {
    if (!worker || worker->last_easgd_socket < 0) {
        printf("[STAGE1-%d] Invalid worker or socket\n", worker->worker_id);
        return -1;
    }
    
    // Flatten weights
    if (flatten_stage1_weights(worker) < 0) {
        printf("[STAGE1-%d] Failed to flatten weights\n", worker->worker_id);
        return -1;
    }
    
    // Create EASGD message
    EASGDMessage* msg = create_easgd_message(EASGD_WEIGHT_UPDATE, worker->worker_id, 1);
    if (!msg) return -1;
    
    msg->epoch = epoch;
    msg->stage1_weight_count = worker->stage1_weight_count;
    msg->stage1_weights = (double*)malloc(sizeof(double) * worker->stage1_weight_count);
    
    if (!msg->stage1_weights) {
        free_easgd_message(msg);
        return -1;
    }
    
    memcpy(msg->stage1_weights, worker->stage1_weights_flat, 
           sizeof(double) * worker->stage1_weight_count);
    
    // Send using the connected socket
    printf("[STAGE1-%d] Sending %d weights to coordinator...\n", 
           worker->worker_id, worker->stage1_weight_count);
    
    if (send_easgd_message(worker->last_easgd_socket, msg) < 0) {
        printf("[STAGE1-%d] Failed to send weights to parameter server\n", worker->worker_id);
        free_easgd_message(msg);
        return -1;
    }
    
    printf("[STAGE1-%d] Weights sent successfully\n", worker->worker_id);
    free_easgd_message(msg);
    return 0;
}

// === FIX receive_master_stage1_weights function ===
int receive_master_stage1_weights(Stage1Worker* worker) {
    if (!worker || worker->last_easgd_socket < 0) {
        printf("[STAGE1-%d] Invalid worker or socket\n", worker->worker_id);
        return -1;
    }
    
    printf("[STAGE1-%d] Waiting for master weights from coordinator...\n", worker->worker_id);
    
    // DON'T close and reconnect, use same socket
    EASGDMessage* msg = receive_easgd_message(worker->last_easgd_socket);
    
    if (!msg || msg->message_type != EASGD_MASTER_WEIGHTS) {
        printf("[STAGE1-%d] Failed to receive master weights\n", worker->worker_id);
        if (msg) free_easgd_message(msg);
        socket_close(worker->last_easgd_socket);
        worker->last_easgd_socket = -1;
        return -1;
    }
    
    printf("[STAGE1-%d] Received %d master weights\n", 
           worker->worker_id, msg->stage1_weight_count);
    
    if (apply_elastic_averaging_stage1(worker, msg->stage1_weights) < 0) {
        printf("[STAGE1-%d] Failed to apply elastic averaging\n", worker->worker_id);
        free_easgd_message(msg);
        socket_close(worker->last_easgd_socket);
        worker->last_easgd_socket = -1;
        return -1;
    }
    
    printf("[STAGE1-%d] Applied elastic averaging successfully\n", worker->worker_id);
    free_easgd_message(msg);
    socket_close(worker->last_easgd_socket);
    worker->last_easgd_socket = -1;
    return 0;
}

void cleanup_stage1_worker(Stage1Worker* worker) {
    if (!worker) return;
    
    if (worker->stage1) free_stage(worker->stage1);
    if (worker->stage1_weights_flat) free(worker->stage1_weights_flat);
    //if (worker->param_server_socket >= 0) socket_close(worker->param_server_socket);
    if (worker->stage2_forward_socket >= 0) socket_close(worker->stage2_forward_socket);
    if (worker->stage2_backward_socket >= 0) socket_close(worker->stage2_backward_socket);
    if (worker->stage2_backward_server >= 0) socket_close(worker->stage2_backward_server);
    if (worker->last_easgd_socket >= 0) socket_close(worker->last_easgd_socket);
    
    cleanup_buffer(&worker->pipeline_buffer);
    cleanup_batch_tracker(&worker->batch_tracker);
    
    free(worker);
}