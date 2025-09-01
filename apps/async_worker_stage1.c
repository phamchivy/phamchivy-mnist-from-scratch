// async_worker_stage1.c - Complete socket-based async with queues
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>
#include "../neural/nn.h"
#include "../neural/pipeline_queue.h"
#include "../neural/batch_storage.h"
#include "../util/img.h"
#include "../socket/socket_utils.h"
#include "../socket/pipeline_socket.h"
#include "../config/config_loader.h"

// Global context for signal handling
static volatile bool shutdown_requested = false;
static ActivationQueue* global_activation_queue = NULL;
static GradientQueue* global_gradient_queue = NULL;

void signal_handler(int sig) {
    shutdown_requested = true;
    printf("\n[Async Stage1] Shutdown signal received\n");
    if (global_activation_queue) activation_queue_shutdown(global_activation_queue);
    if (global_gradient_queue) gradient_queue_shutdown(global_gradient_queue);
}

// Socket sender thread - sends activations from queue to Stage 2
void* socket_sender_thread(void* arg) {
    AsyncPipelineContext* ctx = (AsyncPipelineContext*)arg;
    HybridConfig* config = ctx->config;
    int group_id = ctx->group_id;
    
    // Extract connection info from context (stored in config pointer hack)
    char next_stage_ip[64];
    strcpy(next_stage_ip, "172.32.0.12"); // Group 1 -> worker1_stage2
    if (group_id == 2) {
        strcpy(next_stage_ip, "172.32.0.14"); // Group 2 -> worker2_stage2
    }
    int next_stage_port = 13000 + group_id; // 13001 or 13002
    
    printf("[Socket Sender Group %d] Thread started, sending to %s:%d\n", 
           group_id, next_stage_ip, next_stage_port);
    
    int sent_count = 0;
    
    while (!shutdown_requested) {
        // Dequeue activation from queue
        ActivationMessage activation_msg;
        bool got_activation = activation_queue_dequeue(ctx->activation_queue, 
                                                     &activation_msg, 
                                                     100); // 1 second timeout
        
        if (!got_activation) {
            // Check if queue is shutdown
            pthread_mutex_lock(&ctx->activation_queue->mutex);
            bool queue_shutdown = ctx->activation_queue->shutdown;
            int queue_size = ctx->activation_queue->size;
            pthread_mutex_unlock(&ctx->activation_queue->mutex);
            
            if (queue_shutdown && queue_size == 0) {
                printf("[Socket Sender Group %d] Queue shutdown, no more activations\n", group_id);
                break;
            }
            continue; // Timeout, try again
        }
        
        // Send activation to Stage 2 via socket
        int stage2_sock = connect_to_next_stage(next_stage_ip, next_stage_port);
        if (stage2_sock < 0) {
            printf("[Socket Sender Group %d] Failed to connect to Stage 2\n", group_id);
            matrix_free(activation_msg.activation);
            continue;
        }
        
        // Send batch_id, label, and activation
        if (send_all(stage2_sock, &activation_msg.batch_id, sizeof(int)) != sizeof(int) ||
            send_all(stage2_sock, &activation_msg.label, sizeof(int)) != sizeof(int) ||
            send_activation(stage2_sock, activation_msg.activation) != 0) {
            printf("[Socket Sender Group %d] Failed to send activation for batch %d\n", 
                   group_id, activation_msg.batch_id);
            close(stage2_sock);
            matrix_free(activation_msg.activation);
            continue;
        }
        
        // Receive gradient response
        Matrix* grad_from_stage2 = receive_gradient(stage2_sock, config->network.hidden_size, 1);
        close(stage2_sock);
        
        if (grad_from_stage2) {
            // Enqueue gradient for processing
            double loss = 0.0; // Loss will be calculated in Stage 1 backward
            bool gradient_enqueued = gradient_queue_enqueue(ctx->gradient_queue, 
                                                          grad_from_stage2, 
                                                          activation_msg.batch_id, 
                                                          loss);
            if (!gradient_enqueued) {
                printf("[Socket Sender Group %d] Failed to enqueue gradient for batch %d\n",
                       group_id, activation_msg.batch_id);
                matrix_free(grad_from_stage2);
            }
        } else {
            printf("[Socket Sender Group %d] Failed to receive gradient for batch %d\n",
                   group_id, activation_msg.batch_id);
        }
        
        // Cleanup activation message
        matrix_free(activation_msg.activation);
        sent_count++;
        
        if (sent_count % 1000 == 0) {
            printf("[Socket Sender Group %d] Sent %d activations\n", group_id, sent_count);
        }
    }
    
    printf("[Socket Sender Group %d] Thread completed, sent %d total activations\n", 
           group_id, sent_count);
    return NULL;
}

// Main worker thread - processes data and fills activation queue
void* async_stage1_worker(void* arg) {
    AsyncPipelineContext* ctx = (AsyncPipelineContext*)arg;
    PipelineStage* stage1 = ctx->stage1;
    HybridConfig* config = ctx->config;
    int group_id = ctx->group_id;
    
    printf("[Async Stage1 Worker Group %d] Thread started\n", group_id);
    
    // Create batch storage for gradient matching
    BatchStorage* batch_storage = batch_storage_create();
    if (!batch_storage) {
        printf("[Async Stage1 Group %d] Failed to create batch storage\n", group_id);
        return NULL;
    }
    
    // Load training data
    Img** imgs = csv_to_imgs(config->data.train_path, config->training.total_images);
    if (!imgs) {
        printf("[Async Stage1 Group %d] Failed to load training data\n", group_id);
        batch_storage_destroy(batch_storage);
        return NULL;
    }
    
    // Data partitioning based on group
    int images_per_group = config->training.total_images / config->pipeline.num_groups;
    int start_index = (group_id - 1) * images_per_group;
    int end_index = group_id * images_per_group;
    
    int processed_count = 0;
    int sync_count = 0;
    int gradients_applied = 0;
    double loss_sum = 0.0;
    double start_time = get_timestamp_us() / 1000000.0;
    
    printf("[Async Stage1 Group %d] Processing images %d to %d (%d total)\n", 
           group_id, start_index, end_index-1, end_index - start_index);
    
    for (int epoch = 0; epoch < config->training.epochs && !shutdown_requested; epoch++) {
        printf("[Async Stage1 Group %d] Starting epoch %d/%d\n", 
               group_id, epoch + 1, config->training.epochs);
        
        for (int i = start_index; i < end_index && !shutdown_requested; i++) {
            Img* cur_img = imgs[i];
            Matrix* input = matrix_flatten(cur_img->img_data, 0);
            
            // Forward pass through stage 1
            long long forward_start = get_timestamp_us();
            Matrix* hidden_activation = pipeline_stage1_forward(stage1, input);
            long long forward_end = get_timestamp_us();
            
            int batch_id = processed_count + (epoch * images_per_group);
            
            // Store batch data for later gradient application
            bool stored = batch_storage_store(batch_storage, batch_id, input, hidden_activation);
            if (!stored) {
                printf("[Async Stage1 Group %d] Warning: Failed to store batch %d\n",
                       group_id, batch_id);
            }
            
            // Enqueue activation for socket sender thread
            bool enqueued = activation_queue_enqueue_timeout(ctx->activation_queue, 
                                                           hidden_activation, 
                                                           cur_img->label, 
                                                           batch_id,
                                                           config->async_pipeline.timeout_ms);
            
            if (!enqueued) {
                printf("[Async Stage1 Group %d] Failed to enqueue activation for batch %d\n",
                       group_id, batch_id);
                matrix_free(input);
                matrix_free(hidden_activation);
                continue;
            }
            
            // Process available gradients (non-blocking, multiple per iteration)
            int gradients_processed_this_iter = 0;
            const int MAX_GRADIENTS_PER_ITER = 3;
            
            for (int g = 0; g < MAX_GRADIENTS_PER_ITER; g++) {
                GradientMessage grad_msg;
                bool gradient_available = gradient_queue_dequeue(ctx->gradient_queue, 
                                                               &grad_msg, 
                                                               1); // 1ms timeout
                
                if (!gradient_available) break; // No more gradients available
                
                // Retrieve stored batch data for this gradient
                Matrix* stored_input;
                Matrix* stored_hidden_output;
                bool retrieved = batch_storage_retrieve(batch_storage, grad_msg.batch_id,
                                                      &stored_input, &stored_hidden_output);
                
                if (retrieved) {
                    // Apply gradient to Stage 1 using stored data
                    long long backward_start = get_timestamp_us();
                    double batch_loss = pipeline_stage1_backward(stage1, stored_input, 
                                                               stored_hidden_output, 
                                                               grad_msg.gradient);
                    long long backward_end = get_timestamp_us();
                    
                    // Accumulate loss and statistics
                    loss_sum += grad_msg.loss;
                    gradients_applied++;
                    gradients_processed_this_iter++;
                    
                    if (true && grad_msg.batch_id % 100 == 0 ) { // Fix cứng tham số async_pipeline.enable_profiling == true
                        long long latency = get_timestamp_us() - grad_msg.timestamp;
                        printf("[Async Stage1 Group %d] Applied gradient for batch %d "
                               "(latency: %.2fms, backward: %.2fms)\n",
                               group_id, grad_msg.batch_id,
                               latency / 1000.0, (backward_end - backward_start) / 1000.0);
                    }
                    
                    // Cleanup
                    matrix_free(stored_input);
                    matrix_free(stored_hidden_output);
                } else {
                    printf("[Async Stage1 Group %d] Warning: Could not retrieve batch %d for gradient\n",
                           group_id, grad_msg.batch_id);
                }
                
                matrix_free(grad_msg.gradient);
            }
            
            // Cleanup current iteration
            matrix_free(input);
            matrix_free(hidden_activation);
            processed_count++;
            ctx->samples_processed++;
            
            // Periodic cleanup of old batches
            if (processed_count % 100 == 0) {
                batch_storage_cleanup_old(batch_storage, 10000000); // 10 seconds
            }
            
            // Sync with parameter server every sync_frequency images
            if (processed_count % config->pipeline.sync_frequency == 0) {
                double avg_loss = (gradients_applied > 0) ? loss_sum / gradients_applied : 0.0;
                
                if (config->logging.log_loss) {
                    printf("[Async Stage1 Group %d] Syncing after %d images (sync #%d), "
                           "gradients_applied=%d, avg_loss=%.6f\n", 
                           group_id, processed_count, ++sync_count, gradients_applied, avg_loss);
                } else {
                    printf("[Async Stage1 Group %d] Syncing after %d images (sync #%d)\n", 
                           group_id, processed_count, ++sync_count);
                }
                
                // Get stage 1 weights
                int weight_count;
                double* weights = pipeline_stage1_get_weights(stage1, &weight_count);
                
                // Send to parameter server
                if (send_weights_to_parameter_server(config->server.ip, 12345, 
                                                   group_id, WEIGHT_TYPE_HIDDEN, 
                                                   weights, weight_count) == 0) {
                    if (config->logging.log_sync_details) {
                        printf("[Async Stage1 Group %d] Successfully synced weights\n", group_id);
                    }
                } else {
                    printf("[Async Stage1 Group %d] Failed to sync weights\n", group_id);
                }
                
                free(weights);
                
                // Reset accumulators
                loss_sum = 0.0;
                gradients_applied = 0;
                
                // Print queue and batch storage statistics
                if (true) { // Fix cứng tham số async_pipeline.enable_profiling == true
                    print_queue_stats(ctx->activation_queue, ctx->gradient_queue);
                    batch_storage_print_stats(batch_storage);
                }
            }
            
            // Progress logging
            if (processed_count % 1000 == 0) {
                double elapsed = (get_timestamp_us() / 1000000.0) - start_time;
                double throughput = processed_count / elapsed;
                printf("[Async Stage1 Group %d] Processed %d images (%.1f imgs/sec), "
                       "gradients_applied=%d\n", 
                       group_id, processed_count, throughput, gradients_applied);
            }
        }
        
        printf("[Async Stage1 Group %d] Epoch %d completed\n", group_id, epoch + 1);
    }
    
    // Final gradient processing - drain the gradient queue
    printf("[Async Stage1 Group %d] Draining remaining gradients...\n", group_id);
    int remaining_gradients = 0;
    
    while (true) {
        GradientMessage grad_msg;
        bool gradient_available = gradient_queue_dequeue(ctx->gradient_queue, 
                                                       &grad_msg, 
                                                       1000); // 100ms timeout
        
        if (!gradient_available) break;
        
        Matrix* stored_input;
        Matrix* stored_hidden_output;
        bool retrieved = batch_storage_retrieve(batch_storage, grad_msg.batch_id,
                                              &stored_input, &stored_hidden_output);
        
        if (retrieved) {
            pipeline_stage1_backward(stage1, stored_input, stored_hidden_output, grad_msg.gradient);
            matrix_free(stored_input);
            matrix_free(stored_hidden_output);
            remaining_gradients++;
        }
        
        matrix_free(grad_msg.gradient);
    }
    
    double end_time = get_timestamp_us() / 1000000.0;
    printf("[Async Stage1 Group %d] Worker completed in %.2f seconds\n", 
           group_id, end_time - start_time);
    printf("[Async Stage1 Group %d] Total syncs: %d, remaining gradients processed: %d\n",
           group_id, sync_count, remaining_gradients);
    
    // Signal shutdown to socket sender
    activation_queue_shutdown(ctx->activation_queue);
    
    // Final cleanup
    batch_storage_destroy(batch_storage);
    imgs_free(imgs, config->training.total_images);
    
    return NULL;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        printf("Usage: %s <config_file> <group_id> <next_stage_ip> <next_stage_port> <server_ip> <server_port>\n", argv[0]);
        return 1;
    }
    
    // Setup signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Load configuration
    HybridConfig* config = load_config(argv[1]);
    int group_id = atoi(argv[2]);
    char next_stage_ip[64], server_ip[64];
    strcpy(next_stage_ip, argv[3]);
    int next_stage_port = atoi(argv[4]);
    strcpy(server_ip, argv[5]);
    int server_port = atoi(argv[6]);
    
    if (!config) {
        printf("Failed to load configuration\n");
        return 1;
    }
    
    printf("[Async Stage1 Group %d] Starting socket-based async pipeline with queues\n", group_id);
    printf("[Async Stage1 Group %d] Queue size: %d, Timeout: %dms\n", 
           group_id, config->async_pipeline.queue_size, config->async_pipeline.timeout_ms);
    printf("[Async Stage1 Group %d] Next stage: %s:%d, Server: %s:%d\n", 
           group_id, next_stage_ip, next_stage_port, server_ip, server_port);
    
    srand(time(NULL) + group_id);
    
    // Create async pipeline context with queues
    AsyncPipelineContext* async_ctx = async_pipeline_create(config->async_pipeline.queue_size, group_id);
    if (!async_ctx) {
        printf("[Async Stage1 Group %d] Failed to create async pipeline context\n", group_id);
        free_config(config);
        return 1;
    }
    
    // Store configuration and connection info
    async_ctx->config = config;
    global_activation_queue = async_ctx->activation_queue;
    global_gradient_queue = async_ctx->gradient_queue;
    
    // Initialize pipeline stage 1
    PipelineStage* stage1 = pipeline_stage_create(config->network.input_size, 
                                                 config->network.hidden_size, 
                                                 config->training.learning_rate);
    stage1->easgd_enabled = config->easgd.enabled;
    stage1->alpha = config->easgd.alpha;
    stage1->beta = config->easgd.beta;
    
    async_ctx->stage1 = stage1;
    
    printf("[Async Stage1 Group %d] Pipeline stage initialized: %d→%d, lr=%.3f\n", 
           group_id, config->network.input_size, config->network.hidden_size,
           config->training.learning_rate);
    
    // Start socket sender thread
    pthread_t sender_thread;
    if (pthread_create(&sender_thread, NULL, socket_sender_thread, async_ctx) != 0) {
        printf("[Async Stage1 Group %d] Failed to create socket sender thread\n", group_id);
        async_pipeline_destroy(async_ctx);
        pipeline_stage_free(stage1);
        free_config(config);
        return 1;
    }
    
    // Start main worker thread
    pthread_t worker_thread;
    if (pthread_create(&worker_thread, NULL, async_stage1_worker, async_ctx) != 0) {
        printf("[Async Stage1 Group %d] Failed to create worker thread\n", group_id);
        async_pipeline_destroy(async_ctx);
        pipeline_stage_free(stage1);
        free_config(config);
        return 1;
    }
    
    printf("[Async Stage1 Group %d] Both threads started successfully\n", group_id);
    
    // Main monitoring loop
    while (!shutdown_requested) {
        sleep(1);
        
        // Check queue health and backpressure
        pthread_mutex_lock(&async_ctx->activation_queue->mutex);
        double activation_utilization = (double)async_ctx->activation_queue->size / 
                                      async_ctx->activation_queue->capacity;
        pthread_mutex_unlock(&async_ctx->activation_queue->mutex);
        
        pthread_mutex_lock(&async_ctx->gradient_queue->mutex);
        double gradient_utilization = (double)async_ctx->gradient_queue->size / 
                                    async_ctx->gradient_queue->capacity;
        pthread_mutex_unlock(&async_ctx->gradient_queue->mutex);
        
        // Backpressure warnings
        if (activation_utilization > config->async_pipeline.backpressure_threshold) {
            printf("[Async Stage1 Group %d] Warning: Activation queue %.1f%% full (backpressure)\n",
                   group_id, activation_utilization * 100);
        }
        
        if (gradient_utilization > config->async_pipeline.backpressure_threshold) {
            printf("[Async Stage1 Group %d] Info: Gradient queue %.1f%% full\n",
                   group_id, gradient_utilization * 100);
        }
    }
    
    // Graceful shutdown
    printf("[Async Stage1 Group %d] Shutting down...\n", group_id);
    
    // Signal shutdown and wait for threads
    pthread_join(worker_thread, NULL);
    pthread_join(sender_thread, NULL);
    
    // Save final model if configured
    if (config->model.save_individual_stages) {
        pipeline_stage_save(stage1, "async_stage1", group_id);
        printf("[Async Stage1 Group %d] Final model saved\n", group_id);
    }
    
    // Cleanup
    async_pipeline_destroy(async_ctx);
    pipeline_stage_free(stage1);
    free_config(config);
    
    printf("[Async Stage1 Group %d] Shutdown completed\n", group_id);
    return 0;
}