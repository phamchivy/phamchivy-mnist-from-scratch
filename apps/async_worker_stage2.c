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

// ✅ Proper worker context structure
typedef struct {
    AsyncPipelineContext* pipeline_ctx;
    HybridConfig* config;
    PipelineStage* stage2;
    int group_id;
} Stage2WorkerContext;

// Global context for signal handling
static Stage2WorkerContext* global_worker_ctx = NULL;
static volatile bool shutdown_requested = false;

void signal_handler(int sig) {
    shutdown_requested = true;
    printf("\n[Async Stage2] Shutdown signal received\n");
    if (global_worker_ctx && global_worker_ctx->pipeline_ctx) {
        activation_queue_shutdown(global_worker_ctx->pipeline_ctx->activation_queue);
        gradient_queue_shutdown(global_worker_ctx->pipeline_ctx->gradient_queue);
    }
}

// ✅ Fixed async Stage 2 worker thread
void* async_stage2_worker(void* arg) {
    Stage2WorkerContext* worker_ctx = (Stage2WorkerContext*)arg;
    AsyncPipelineContext* ctx = worker_ctx->pipeline_ctx;
    HybridConfig* config = worker_ctx->config;
    PipelineStage* stage2 = worker_ctx->stage2;
    int group_id = worker_ctx->group_id;
    
    printf("[Async Stage2 Group %d] Worker thread started\n", group_id);
    
    int processed_count = 0;
    int sync_count = 0;
    double loss_sum = 0.0;
    double start_time = get_timestamp_us() / 1000000.0;
    
    // Calculate target images per group
    int images_per_group = config->training.total_images / config->pipeline.num_groups;
    
    printf("[Async Stage2 Group %d] Processing up to %d images\n", 
           group_id, images_per_group);
    
    while (processed_count < images_per_group && !shutdown_requested) {
        // ✅ Dequeue activation from Stage 1 with timeout
        ActivationMessage activation_msg;
        bool activation_available = activation_queue_dequeue(ctx->activation_queue, 
                                                           &activation_msg, 
                                                           config->async_pipeline.timeout_ms);
        
        if (!activation_available) {
            // ✅ Check if Stage 1 has completed and queue is empty
            pthread_mutex_lock(&ctx->activation_queue->mutex);
            bool queue_shutdown = ctx->activation_queue->shutdown;
            int queue_size = ctx->activation_queue->size;
            pthread_mutex_unlock(&ctx->activation_queue->mutex);
            
            if (queue_shutdown && queue_size == 0) {
                printf("[Async Stage2 Group %d] Stage1 completed, no more activations\n", group_id);
                break;
            }
            
            // Timeout occurred, continue waiting
            if (config->async_pipeline.enable_profiling) {
                printf("[Async Stage2 Group %d] Waiting for activations (timeout)\n", group_id);
            }
            continue;
        }
        
        // ✅ Validate received activation
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
        
        // ✅ Create target output (one-hot encoding)
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
        
        // ✅ Forward pass through stage 2
        Matrix* final_outputs = pipeline_stage2_forward(stage2, activation_msg.activation);
        if (!final_outputs) {
            printf("[Async Stage2 Group %d] Forward pass failed for batch %d\n", 
                   group_id, activation_msg.batch_id);
            matrix_free(activation_msg.activation);
            matrix_free(target);
            continue;
        }
        
        // ✅ Backward pass through stage 2
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
        ctx->samples_processed++;
        
        // ✅ Calculate processing latency
        long long latency_us = stage2_end - activation_msg.timestamp;
        double latency_ms = latency_us / 1000.0;
        double stage2_time_ms = (stage2_end - stage2_start) / 1000.0;
        
        if (config->async_pipeline.enable_profiling && processed_count % 100 == 0) {
            printf("[Async Stage2 Group %d] Batch %d: loss=%.6f, latency=%.2fms, stage2_time=%.2fms\n",
                   group_id, activation_msg.batch_id, loss, latency_ms, stage2_time_ms);
        }
        
        // ✅ Enqueue gradient for Stage 1 (non-blocking with timeout)
        bool gradient_enqueued = gradient_queue_enqueue(ctx->gradient_queue, 
                                                      grad_to_stage1, 
                                                      activation_msg.batch_id, 
                                                      loss);
        
        if (!gradient_enqueued) {
            printf("[Async Stage2 Group %d] Warning: Failed to enqueue gradient for batch %d\n",
                   group_id, activation_msg.batch_id);
        }
        
        // ✅ Cleanup current iteration
        matrix_free(activation_msg.activation);
        matrix_free(target);
        matrix_free(final_outputs);
        // Note: grad_to_stage1 is now owned by gradient queue
        
        // ✅ Periodic sync with parameter server
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
                if (send_weights_to_parameter_server(config->server.ip, 12345, 
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
            if (config->async_pipeline.enable_profiling) {
                print_queue_stats(ctx->activation_queue, ctx->gradient_queue);
                //batch_storage_print_stats(batch_storage);
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
    
    // ✅ Signal shutdown to gradient consumers (Stage1)
    gradient_queue_shutdown(ctx->gradient_queue);
    
    printf("[Async Stage2 Group %d] Processed %d total images\n", group_id, processed_count);
    return NULL;
}

int main(int argc, char** argv) {
    // ✅ IMMEDIATE DEBUG LOGGING
    printf("[DEBUG Stage2] Starting with %d arguments\n", argc);
    for (int i = 0; i < argc; i++) {
        printf("[DEBUG Stage2] argv[%d] = '%s'\n", i, argv[i]);
    }
    fflush(stdout);

    if (argc < 5) {
        printf("[ERROR Stage2] Not enough arguments!\n");
        printf("Usage: %s <config_file> <group_id> <listen_port> <server_ip> <server_port>\n", argv[0]);
        printf("   OR: %s <group_id> <listen_port> <server_ip> <server_port> (uses default config.yml)\n", argv[0]);
        fflush(stdout);
        return 1;
    }

    // Setup signal handlers
    printf("[DEBUG Stage2] Setting up signal handlers...\n");
    fflush(stdout);
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // ✅ Load configuration with error handling
    printf("[DEBUG Stage2] Loading configuration...\n");
    fflush(stdout);
    
    HybridConfig* config;
    int group_id, listen_port, server_port;
    char server_ip[64];
    
    if (argc == 5) {
        printf("[DEBUG Stage2] Using old format with default config.yml\n");
        fflush(stdout);
        config = load_config("config.yml");
        group_id = atoi(argv[1]);
        listen_port = atoi(argv[2]);
        strcpy(server_ip, argv[3]);
        server_port = atoi(argv[4]);
    } else {
        printf("[DEBUG Stage2] Using new format with config file: %s\n", argv[1]);
        fflush(stdout);
        config = load_config(argv[1]);
        group_id = atoi(argv[2]);
        listen_port = atoi(argv[3]);
        strcpy(server_ip, argv[4]);
        server_port = atoi(argv[5]);
    }
    
    if (!config) {
        printf("[ERROR Stage2] Failed to load configuration!\n");
        fflush(stdout);
        return 1;
    }

    printf("[DEBUG Stage2] Configuration loaded successfully\n");
    printf("[DEBUG Stage2] Group ID: %d, Listen port: %d\n", group_id, listen_port);
    printf("[DEBUG Stage2] Server: %s:%d\n", server_ip, server_port);
    fflush(stdout);
    
    printf("[Async Stage2 Group %d] Starting with asynchronous pipeline\n", group_id);
    printf("[Async Stage2 Group %d] Config: queue_size=%d, timeout=%dms, profiling=%s\n", 
           group_id, config->async_pipeline.queue_size, config->async_pipeline.timeout_ms,
           config->async_pipeline.enable_profiling ? "enabled" : "disabled");
    
    // Add detailed debug logging trong async_worker_stage2.c

    printf("[DEBUG Stage2] About to create shared memory key...\n");
    fflush(stdout);

    char shared_memory_key[64];
    printf("[DEBUG Stage2] Calling snprintf for shared memory key...\n");
    fflush(stdout);

    snprintf(shared_memory_key, sizeof(shared_memory_key), "/async_pipeline_group_%d", group_id);

    printf("[DEBUG Stage2] Shared memory key created: '%s'\n", shared_memory_key);
    fflush(stdout);

    printf("[DEBUG Stage2] About to call async_pipeline_create_shared...\n");
    fflush(stdout);

    AsyncPipelineContext* async_ctx = async_pipeline_create_shared(config->async_pipeline.queue_size, 
                                                                group_id, 
                                                                shared_memory_key);

    printf("[DEBUG Stage2] async_pipeline_create_shared returned: %p\n", (void*)async_ctx);
    fflush(stdout);

    if (!async_ctx) {
        printf("[ERROR Stage2] Failed to create shared async pipeline context\n");
        fflush(stdout);
        free_config(config);
        return 1;
    }

    printf("[DEBUG Stage2] About to create pipeline stage 2...\n");
    fflush(stdout);

    PipelineStage* stage2 = pipeline_stage_create(config->network.hidden_size, 
                                                config->network.output_size, 
                                                config->training.learning_rate);

    printf("[DEBUG Stage2] pipeline_stage_create returned: %p\n", (void*)stage2);
    fflush(stdout);

    if (!stage2) {
        printf("[ERROR Stage2] Failed to create pipeline stage\n");
        fflush(stdout);
        async_pipeline_destroy_shared(async_ctx);
        free_config(config);
        return 1;
    }

    printf("[DEBUG Stage2] Setting stage2 EASGD parameters...\n");
    fflush(stdout);

    stage2->easgd_enabled = config->easgd.enabled;
    stage2->alpha = config->easgd.alpha;
    stage2->beta = config->easgd.beta;

    printf("[DEBUG Stage2] Creating worker context...\n");
    fflush(stdout);

    Stage2WorkerContext* worker_ctx = malloc(sizeof(Stage2WorkerContext));

    printf("[DEBUG Stage2] worker_ctx malloc returned: %p\n", (void*)worker_ctx);
    fflush(stdout);

    if (!worker_ctx) {
        printf("[ERROR Stage2] Failed to allocate worker context\n");
        fflush(stdout);
        pipeline_stage_free(stage2);
        async_pipeline_destroy_shared(async_ctx);
        free_config(config);
        return 1;
    }

    printf("[DEBUG Stage2] Setting worker context fields...\n");
    fflush(stdout);

    worker_ctx->pipeline_ctx = async_ctx;
    worker_ctx->config = config;
    worker_ctx->stage2 = stage2;
    worker_ctx->group_id = group_id;
    global_worker_ctx = worker_ctx;

    printf("[DEBUG Stage2] About to call handshake...\n");
    fflush(stdout);

    // Continue with handshake...
    
    // ✅ Connection handshake via shared memory
    printf("[Async Stage2 Group %d] Waiting for Stage1 connection...\n", group_id);
    
    // Set Stage2 ready flag
    async_pipeline_set_stage2_ready(async_ctx);
    
    // Wait for Stage1 ready flag (with timeout)
    int connection_timeout = 30; // 30 seconds
    int waited = 0;
    while (!async_pipeline_is_stage1_ready(async_ctx) && waited < connection_timeout && !shutdown_requested) {
        sleep(1);
        waited++;
        if (waited % 5 == 0) {
            printf("[Async Stage2 Group %d] Still waiting for Stage1... (%d/%d)\n", 
                   group_id, waited, connection_timeout);
        }
    }
    
    if (!async_pipeline_is_stage1_ready(async_ctx)) {
        printf("[Async Stage2 Group %d] ERROR: Stage1 connection timeout\n", group_id);
        free(worker_ctx);
        pipeline_stage_free(stage2);
        async_pipeline_destroy_shared(async_ctx);
        free_config(config);
        return 1;
    }
    
    printf("[Async Stage2 Group %d] Connected to Stage1, starting processing\n", group_id);
    
    // ✅ Start the worker thread
    pthread_t worker_thread;
    if (pthread_create(&worker_thread, NULL, async_stage2_worker, worker_ctx) != 0) {
        printf("[Async Stage2 Group %d] Failed to create worker thread\n", group_id);
        free(worker_ctx);
        pipeline_stage_free(stage2);
        async_pipeline_destroy_shared(async_ctx);
        free_config(config);
        return 1;
    }
    
    // ✅ Main monitoring loop
    while (!shutdown_requested) {
        sleep(1);
        
        // Monitor queue health
        pthread_mutex_lock(&async_ctx->activation_queue->mutex);
        double activation_utilization = (double)async_ctx->activation_queue->size / 
                                      async_ctx->activation_queue->capacity;
        bool activation_shutdown = async_ctx->activation_queue->shutdown;
        int activation_size = async_ctx->activation_queue->size;
        pthread_mutex_unlock(&async_ctx->activation_queue->mutex);
        
        pthread_mutex_lock(&async_ctx->gradient_queue->mutex);
        double gradient_utilization = (double)async_ctx->gradient_queue->size / 
                                    async_ctx->gradient_queue->capacity;
        pthread_mutex_unlock(&async_ctx->gradient_queue->mutex);
        
        // Queue utilization warnings
        if (activation_utilization > config->async_pipeline.backpressure_threshold) {
            printf("[Async Stage2 Group %d] Warning: Activation queue %.1f%% full\n",
                   group_id, activation_utilization * 100);
        }
        
        if (gradient_utilization > config->async_pipeline.backpressure_threshold) {
            printf("[Async Stage2 Group %d] Warning: Gradient queue %.1f%% full (backpressure to Stage1)\n",
                   group_id, gradient_utilization * 100);
        }
        
        // ✅ Check if Stage1 completed and activation queue is empty
        if (activation_shutdown && activation_size == 0) {
            printf("[Async Stage2 Group %d] Stage1 completed and queue empty, initiating shutdown\n", group_id);
            break;
        }
    }
    
    // ✅ Graceful shutdown
    printf("[Async Stage2 Group %d] Shutting down...\n", group_id);
    
    // Signal shutdown and wait for worker thread
    shutdown_requested = true;
    activation_queue_shutdown(async_ctx->activation_queue);
    gradient_queue_shutdown(async_ctx->gradient_queue);
    
    pthread_join(worker_thread, NULL);
    
    // Save final model if configured
    if (config->model.save_individual_stages) {
        pipeline_stage_save(stage2, "async_stage2", group_id);
        printf("[Async Stage2 Group %d] Final model saved\n", group_id);
    }
    
    // ✅ Cleanup
    free(worker_ctx);
    pipeline_stage_free(stage2);
    async_pipeline_destroy_shared(async_ctx);
    free_config(config);
    
    printf("[Async Stage2 Group %d] Shutdown completed\n", group_id);
    return 0;
}