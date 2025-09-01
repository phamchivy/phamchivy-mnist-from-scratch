// async_worker_stage2.c - Complete socket-based async with queues
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>
#include "../neural/nn.h"
#include "../neural/pipeline_queue.h"
#include "../socket/socket_utils.h"
#include "../socket/pipeline_socket.h"
#include "../config/config_loader.h"

typedef struct {
    HybridConfig* config;
    PipelineStage* stage2;
    int group_id;
    int listen_port;
    char server_ip[64];
    int server_port;
    ActivationQueue* activation_queue;
    GradientQueue* gradient_queue;
} Stage2WorkerContext;

// Global context for signal handling
static Stage2WorkerContext* global_worker_ctx = NULL;
static volatile bool shutdown_requested = false;

void signal_handler(int sig) {
    shutdown_requested = true;
    printf("\n[Async Stage2] Shutdown signal received\n");
    if (global_worker_ctx) {
        activation_queue_shutdown(global_worker_ctx->activation_queue);
        gradient_queue_shutdown(global_worker_ctx->gradient_queue);
    }
}

// Socket receiver thread - receives activations and fills queue
void* socket_receiver_thread(void* arg) {
    Stage2WorkerContext* worker_ctx = (Stage2WorkerContext*)arg;
    int group_id = worker_ctx->group_id;
    int listen_port = worker_ctx->listen_port;
    HybridConfig* config = worker_ctx->config;
    
    printf("[Socket Receiver Group %d] Thread started, listening on port %d\n", 
           group_id, listen_port);
    
    // Setup server to receive activations from stage 1
    int server_sock = setup_activation_server(listen_port);
    if (server_sock < 0) {
        printf("[Socket Receiver Group %d] Failed to setup activation server\n", group_id);
        return NULL;
    }
    
    printf("[Socket Receiver Group %d] Activation server ready on port %d\n", 
           group_id, listen_port);
    
    int received_count = 0;
    
    while (!shutdown_requested) {
        // Accept connection from stage 1
        int stage1_sock = accept_client(server_sock);
        if (stage1_sock < 0) {
            if (!shutdown_requested) {
                printf("[Socket Receiver Group %d] Failed to accept connection\n", group_id);
            }
            continue;
        }
        
        // Receive batch_id
        int batch_id;
        if (recv_all(stage1_sock, &batch_id, sizeof(int)) != sizeof(int)) {
            printf("[Socket Receiver Group %d] Failed to receive batch_id\n", group_id);
            close(stage1_sock);
            continue;
        }
        
        // Receive image label
        int label;
        if (recv_all(stage1_sock, &label, sizeof(int)) != sizeof(int)) {
            printf("[Socket Receiver Group %d] Failed to receive label\n", group_id);
            close(stage1_sock);
            continue;
        }
        
        // Receive hidden activation from stage 1
        Matrix* hidden_activation = receive_activation(stage1_sock, config->network.hidden_size, 1);
        if (hidden_activation == NULL) {
            printf("[Socket Receiver Group %d] Failed to receive activation\n", group_id);
            close(stage1_sock);
            continue;
        }
        
        // Validate activation and label
        if (label < 0 || label >= config->network.output_size) {
            printf("[Socket Receiver Group %d] Invalid label %d, skipping batch %d\n", 
                   group_id, label, batch_id);
            matrix_free(hidden_activation);
            close(stage1_sock);
            continue;
        }
        
        // Enqueue activation for processing
        bool enqueued = activation_queue_enqueue_timeout(worker_ctx->activation_queue,
                                                       hidden_activation,
                                                       label,
                                                       batch_id,
                                                       config->async_pipeline.timeout_ms);
        
        if (!enqueued) {
            printf("[Socket Receiver Group %d] Failed to enqueue activation for batch %d\n",
                   group_id, batch_id);
            matrix_free(hidden_activation);
            close(stage1_sock);
            continue;
        }
        
        // Try to get gradient response for this batch_id
        GradientMessage grad_msg;
        bool got_gradient = gradient_queue_dequeue(worker_ctx->gradient_queue,
                                                  &grad_msg,
                                                  config->async_pipeline.timeout_ms);
        
        if (got_gradient) {
            // Send gradient back to stage 1
            if (send_gradient(stage1_sock, grad_msg.gradient) == 0) {
                if (true && received_count % 100 == 0) { //Fix cứng tham số async_pipeline.enable_profiling == true
                    printf("[Socket Receiver Group %d] Sent gradient for batch %d\n",
                           group_id, grad_msg.batch_id);
                }
            } else {
                printf("[Socket Receiver Group %d] Failed to send gradient for batch %d\n",
                       group_id, grad_msg.batch_id);
            }
            matrix_free(grad_msg.gradient);
        } else {
            // No gradient available yet, send zero gradient or close connection
            printf("[Socket Receiver Group %d] No gradient available for batch %d\n",
                   group_id, batch_id);
        }
        
        close(stage1_sock);
        received_count++;
        
        if (received_count % 1000 == 0) {
            printf("[Socket Receiver Group %d] Received %d activations\n", 
                   group_id, received_count);
        }
    }
    
    close(server_sock);
    
    printf("[Socket Receiver Group %d] Thread completed, received %d total activations\n",
           group_id, received_count);
    return NULL;
}

// Processing worker thread - processes activations from queue
void* async_stage2_worker(void* arg) {
    Stage2WorkerContext* worker_ctx = (Stage2WorkerContext*)arg;
    HybridConfig* config = worker_ctx->config;
    PipelineStage* stage2 = worker_ctx->stage2;
    int group_id = worker_ctx->group_id;
    
    printf("[Async Stage2 Worker Group %d] Thread started\n", group_id);
    
    int processed_count = 0;
    int sync_count = 0;
    double loss_sum = 0.0;
    double start_time = get_timestamp_us() / 1000000.0;
    
    // Calculate target images per group
    int images_per_group = config->training.total_images / config->pipeline.num_groups;
    
    printf("[Async Stage2 Group %d] Processing up to %d images\n", 
           group_id, images_per_group);
    
    while (processed_count < images_per_group && !shutdown_requested) {
        // Dequeue activation from queue
        ActivationMessage activation_msg;
        bool activation_available = activation_queue_dequeue(worker_ctx->activation_queue, 
                                                           &activation_msg, 
                                                           config->async_pipeline.timeout_ms);
        
        if (!activation_available) {
            // Check if queue is shutdown and empty
            pthread_mutex_lock(&worker_ctx->activation_queue->mutex);
            bool queue_shutdown = worker_ctx->activation_queue->shutdown;
            int queue_size = worker_ctx->activation_queue->size;
            pthread_mutex_unlock(&worker_ctx->activation_queue->mutex);
            
            if (queue_shutdown && queue_size == 0) {
                printf("[Async Stage2 Group %d] Activation queue shutdown, no more activations\n", group_id);
                break;
            }
            
            // Timeout occurred, continue waiting
            if (true) { //Fix cứng tham số async_pipeline.enable_profiling == true
                printf("[Async Stage2 Group %d] Waiting for activations (timeout)\n", group_id);
            }
            continue;
        }
        
        // Validate received activation
        if (!activation_msg.activation) {
            printf("[Async Stage2 Group %d] Received null activation, skipping\n", group_id);
            continue;
        }
        
        if (activation_msg.label < 0 || activation_msg.label >= config->network.output_size) {
            printf("[Async Stage2 Group %d] Invalid label %d, skipping batch %d\n", 
                   group_id, activation_msg.label, activation_msg.batch_id);
            matrix_free(activation_msg.activation);
            continue;
        }
        
        // Validate activation dimensions
        if (activation_msg.activation->rows != config->network.hidden_size || 
            activation_msg.activation->cols != 1) {
            printf("[Async Stage2 Group %d] Invalid activation dimensions: %dx%d (expected %dx1)\n", 
                   group_id, activation_msg.activation->rows, activation_msg.activation->cols,
                   config->network.hidden_size);
            matrix_free(activation_msg.activation);
            continue;
        }
        
        long long stage2_start = get_timestamp_us();
        
        // Create target output (one-hot encoding)
        Matrix* target = matrix_create(config->network.output_size, 1);
        if (!target) {
            printf("[Async Stage2 Group %d] Failed to create target matrix\n", group_id);
            matrix_free(activation_msg.activation);
            continue;
        }
        
        // Zero out target and set correct label to 1.0
        for (int i = 0; i < config->network.output_size; i++) {
            target->entries[i][0] = 0.0;
        }
        target->entries[activation_msg.label][0] = 1.0;
        
        // Forward pass through stage 2
        Matrix* final_outputs = pipeline_stage2_forward(stage2, activation_msg.activation);
        if (!final_outputs) {
            printf("[Async Stage2 Group %d] Forward pass failed for batch %d\n", 
                   group_id, activation_msg.batch_id);
            matrix_free(activation_msg.activation);
            matrix_free(target);
            continue;
        }
        
        // Backward pass through stage 2
        Matrix* grad_to_stage1;
        double loss = pipeline_stage2_backward(stage2, activation_msg.activation, 
                                             final_outputs, target, &grad_to_stage1);
        
        if (!grad_to_stage1) {
            printf("[Async Stage2 Group %d] Backward pass failed for batch %d\n",
                   group_id, activation_msg.batch_id);
            matrix_free(activation_msg.activation);
            matrix_free(target);
            matrix_free(final_outputs);
            continue;
        }
        
        long long stage2_end = get_timestamp_us();
        
        loss_sum += loss;
        processed_count++;
        
        // Calculate processing latency
        long long latency_us = stage2_end - activation_msg.timestamp;
        double latency_ms = latency_us / 1000.0;
        double stage2_time_ms = (stage2_end - stage2_start) / 1000.0;
        
        if (true && processed_count % 100 == 0) { // Fix cứng tham số async_pipeline.enable_profiling == true
            printf("[Async Stage2 Group %d] Batch %d: loss=%.6f, latency=%.2fms, stage2_time=%.2fms\n",
                   group_id, activation_msg.batch_id, loss, latency_ms, stage2_time_ms);
        }
        
        // Enqueue gradient for socket sender
        bool gradient_enqueued = gradient_queue_enqueue(worker_ctx->gradient_queue, 
                                                      grad_to_stage1, 
                                                      activation_msg.batch_id, 
                                                      loss);
        
        if (!gradient_enqueued) {
            printf("[Async Stage2 Group %d] Warning: Failed to enqueue gradient for batch %d\n",
                   group_id, activation_msg.batch_id);
            matrix_free(grad_to_stage1);
        }
        
        // Cleanup current iteration
        matrix_free(activation_msg.activation);
        matrix_free(target);
        matrix_free(final_outputs);
        // Note: grad_to_stage1 is now owned by gradient queue
        
        // Periodic sync with parameter server
        if (processed_count % config->pipeline.sync_frequency == 0) {
            double avg_loss = loss_sum / config->pipeline.sync_frequency;
            
            if (config->logging.log_loss) {
                printf("[Async Stage2 Group %d] Syncing after %d images (sync #%d), avg_loss=%.6f\n", 
                       group_id, processed_count, ++sync_count, avg_loss);
            } else {
                printf("[Async Stage2 Group %d] Syncing after %d images (sync #%d)\n", 
                       group_id, processed_count, ++sync_count);
            }
            
            // Get stage 2 weights
            int weight_count;
            double* weights = pipeline_stage2_get_weights(stage2, &weight_count);
            
            if (weights) {
                // Send to parameter server
                if (send_weights_to_parameter_server(worker_ctx->server_ip, worker_ctx->server_port, 
                                                   group_id, WEIGHT_TYPE_OUTPUT, 
                                                   weights, weight_count) == 0) {
                    if (config->logging.log_sync_details) {
                        printf("[Async Stage2 Group %d] Successfully synced weights\n", group_id);
                    }
                } else {
                    printf("[Async Stage2 Group %d] Failed to sync weights\n", group_id);
                }
                
                free(weights);
            }
            
            loss_sum = 0.0;
            
            // Print queue statistics
            if (true) { // Fix cứng tham số async_pipeline.enable_profiling == true
                print_queue_stats(worker_ctx->activation_queue, worker_ctx->gradient_queue);
            }
        }
        
        // Progress logging
        if (processed_count % 1000 == 0) {
            double elapsed = (get_timestamp_us() / 1000000.0) - start_time;
            double throughput = processed_count / elapsed;
            printf("[Async Stage2 Group %d] Processed %d images (%.1f imgs/sec)\n", 
                   group_id, processed_count, throughput);
        }
    }
    
    double end_time = get_timestamp_us() / 1000000.0;
    printf("[Async Stage2 Group %d] Worker completed in %.2f seconds, total syncs: %d\n", 
           group_id, end_time - start_time, sync_count);
    
    // Signal shutdown to gradient consumers
    gradient_queue_shutdown(worker_ctx->gradient_queue);
    
    printf("[Async Stage2 Group %d] Processed %d total images\n", group_id, processed_count);
    return NULL;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        printf("Usage: %s <config_file> <group_id> <listen_port> <server_ip> <server_port>\n", argv[0]);
        return 1;
    }
    
    // Setup signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Load configuration
    HybridConfig* config = load_config(argv[1]);
    int group_id = atoi(argv[2]);
    int listen_port = atoi(argv[3]);
    char server_ip[64];
    strcpy(server_ip, argv[4]);
    int server_port = atoi(argv[5]);
    
    if (!config) {
        printf("Failed to load configuration\n");
        return 1;
    }
    
    printf("[Async Stage2 Group %d] Starting socket-based async pipeline with queues\n", group_id);
    printf("[Async Stage2 Group %d] Config: queue_size=%d, timeout=%dms, profiling=%s\n", 
           group_id, config->async_pipeline.queue_size, config->async_pipeline.timeout_ms,
           config->async_pipeline.enable_profiling ? "enabled" : "disabled");
    printf("[Async Stage2 Group %d] Listen port: %d, Server: %s:%d\n", 
           group_id, listen_port, server_ip, server_port);
    
    srand(time(NULL) + group_id + 100);
    
    // Create activation and gradient queues
    ActivationQueue* activation_queue = activation_queue_create(config->async_pipeline.queue_size);
    GradientQueue* gradient_queue = gradient_queue_create(config->async_pipeline.queue_size);
    
    if (!activation_queue || !gradient_queue) {
        printf("[Async Stage2 Group %d] Failed to create queues\n", group_id);
        if (activation_queue) activation_queue_destroy(activation_queue);
        if (gradient_queue) gradient_queue_destroy(gradient_queue);
        free_config(config);
        return 1;
    }
    
    printf("[Async Stage2 Group %d] Queues created successfully\n", group_id);
    
    // Initialize pipeline stage 2
    PipelineStage* stage2 = pipeline_stage_create(config->network.hidden_size, 
                                                 config->network.output_size, 
                                                 config->training.learning_rate);
    if (!stage2) {
        printf("[Async Stage2 Group %d] Failed to create pipeline stage\n", group_id);
        activation_queue_destroy(activation_queue);
        gradient_queue_destroy(gradient_queue);
        free_config(config);
        return 1;
    }
    
    stage2->easgd_enabled = config->easgd.enabled;
    stage2->alpha = config->easgd.alpha;
    stage2->beta = config->easgd.beta;
    
    printf("[Async Stage2 Group %d] Pipeline stage initialized: %d→%d, lr=%.3f\n", 
           group_id, config->network.hidden_size, config->network.output_size,
           config->training.learning_rate);
    
    // Create worker context
    Stage2WorkerContext* worker_ctx = malloc(sizeof(Stage2WorkerContext));
    if (!worker_ctx) {
        printf("[Async Stage2 Group %d] Failed to allocate worker context\n", group_id);
        pipeline_stage_free(stage2);
        activation_queue_destroy(activation_queue);
        gradient_queue_destroy(gradient_queue);
        free_config(config);
        return 1;
    }
    
    worker_ctx->config = config;
    worker_ctx->stage2 = stage2;
    worker_ctx->group_id = group_id;
    worker_ctx->listen_port = listen_port;
    strcpy(worker_ctx->server_ip, server_ip);
    worker_ctx->server_port = server_port;
    worker_ctx->activation_queue = activation_queue;
    worker_ctx->gradient_queue = gradient_queue;
    global_worker_ctx = worker_ctx;
    
    printf("[Async Stage2 Group %d] Starting socket receiver and processing threads\n", group_id);
    
    // Start socket receiver thread
    pthread_t receiver_thread;
    if (pthread_create(&receiver_thread, NULL, socket_receiver_thread, worker_ctx) != 0) {
        printf("[Async Stage2 Group %d] Failed to create socket receiver thread\n", group_id);
        free(worker_ctx);
        pipeline_stage_free(stage2);
        activation_queue_destroy(activation_queue);
        gradient_queue_destroy(gradient_queue);
        free_config(config);
        return 1;
    }
    
    // Start processing worker thread
    pthread_t worker_thread;
    if (pthread_create(&worker_thread, NULL, async_stage2_worker, worker_ctx) != 0) {
        printf("[Async Stage2 Group %d] Failed to create worker thread\n", group_id);
        free(worker_ctx);
        pipeline_stage_free(stage2);
        activation_queue_destroy(activation_queue);
        gradient_queue_destroy(gradient_queue);
        free_config(config);
        return 1;
    }
    
    printf("[Async Stage2 Group %d] Both threads started successfully\n", group_id);
    
    // Main monitoring loop
    while (!shutdown_requested) {
        sleep(1);
        
        // Monitor queue health
        pthread_mutex_lock(&activation_queue->mutex);
        double activation_utilization = (double)activation_queue->size / activation_queue->capacity;
        bool activation_shutdown = activation_queue->shutdown;
        int activation_size = activation_queue->size;
        pthread_mutex_unlock(&activation_queue->mutex);
        
        pthread_mutex_lock(&gradient_queue->mutex);
        double gradient_utilization = (double)gradient_queue->size / gradient_queue->capacity;
        pthread_mutex_unlock(&gradient_queue->mutex);
        
        // Queue utilization warnings
        if (activation_utilization > config->async_pipeline.backpressure_threshold) {
            printf("[Async Stage2 Group %d] Warning: Activation queue %.1f%% full\n",
                   group_id, activation_utilization * 100);
        }
        
        if (gradient_utilization > config->async_pipeline.backpressure_threshold) {
            printf("[Async Stage2 Group %d] Warning: Gradient queue %.1f%% full (backpressure to Stage1)\n",
                   group_id, gradient_utilization * 100);
        }
        
        // Check if processing completed
        if (activation_shutdown && activation_size == 0) {
            printf("[Async Stage2 Group %d] Processing completed, initiating shutdown\n", group_id);
            break;
        }

        // ✅ CHECK for Ctrl+C signal - thoát ngay lập tức
        if (shutdown_requested) {
            printf("[Async Stage2 Group %d] Shutdown signal received\n", group_id);
            break;
        }
    }
    
    // Graceful shutdown
    printf("[Async Stage2 Group %d] Shutting down...\n", group_id);
    
    // Signal shutdown and wait for threads
    shutdown_requested = true;
    activation_queue_shutdown(activation_queue);
    gradient_queue_shutdown(gradient_queue);
    
    pthread_join(receiver_thread, NULL);
    pthread_join(worker_thread, NULL);
    
    // Save final model if configured
    if (config->model.save_individual_stages) {
        pipeline_stage_save(stage2, "async_stage2", group_id);
        printf("[Async Stage2 Group %d] Final model saved\n", group_id);
    }
    
    // Cleanup
    free(worker_ctx);
    pipeline_stage_free(stage2);
    activation_queue_destroy(activation_queue);
    gradient_queue_destroy(gradient_queue);
    free_config(config);
    
    printf("[Async Stage2 Group %d] Shutdown completed\n", group_id);
    return 0;
}