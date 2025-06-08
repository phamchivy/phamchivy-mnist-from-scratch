#include "pipeline_nn.h"
#include "../socket/socket_utils.h"
#include "../util/img.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

// Global variables for timing and stats
static struct timeval start_time, end_time;
static int training_active = 1;
PipelineStats global_stats = {0};
pthread_mutex_t stats_mutex = PTHREAD_MUTEX_INITIALIZER;

double get_time_diff(struct timeval start, struct timeval end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
}

void pipeline_stage1_main(int forward_port, int backward_port) {
    printf("[STAGE1] Starting Stage 1 with Pipeline Optimization (Phase 2+3) on ports %d/%d\n", forward_port, backward_port);
    fflush(stdout);
    
    // Small delay to ensure proper initialization
    sleep(2);
    
    // Load dataset
    int number_imgs = 60000; // Use full MNIST training dataset
    Img** imgs = csv_to_imgs("./data/mnist_train.csv", number_imgs);
    if (!imgs) {
        printf("[STAGE1] Error loading dataset\n");
        return;
    }
    
    // Create stage 1 network (input -> hidden) with proper learning rate
    NetworkStage* stage1 = create_stage(1, 784, 512, 0.01);
    if (!stage1) {
        printf("[STAGE1] Error creating network stage\n");
        imgs_free(imgs, number_imgs);
        return;
    }
    
    // Initialize pipeline buffer and batch tracker
    PipelineBuffer pipeline_buffer;
    if (initialize_buffer(&pipeline_buffer) < 0) {
        printf("[STAGE1] Error initializing pipeline buffer\n");
        free_stage(stage1);
        imgs_free(imgs, number_imgs);
        return;
    }
    
    PipelineBatchTracker batch_tracker;
    if (initialize_batch_tracker(&batch_tracker) < 0) {
        printf("[STAGE1] Error initializing batch tracker\n");
        cleanup_buffer(&pipeline_buffer);
        free_stage(stage1);
        imgs_free(imgs, number_imgs);
        return;
    }
    
    // Initialize adaptive pipeline configuration
    PipelineConfig config = {
        .current_batch_size = MINI_BATCH_SIZE,
        .pipeline_depth = 4,
        .target_latency = 0.1, // 100ms target
        .communication_ratio = 0.0,
        .last_adjustment_time = 0
    };
    
    // Setup sockets
    int forward_server = setup_server(forward_port);
    if (forward_server < 0) {
        printf("[STAGE1] Error setting up forward server\n");
        cleanup_batch_tracker(&batch_tracker);
        cleanup_buffer(&pipeline_buffer);
        free_stage(stage1);
        imgs_free(imgs, number_imgs);
        return;
    }
    
    printf("[STAGE1] Waiting for Stage 2 connection...\n");
    fflush(stdout);
    
    int forward_client = accept_client(forward_server);
    if (forward_client < 0) {
        printf("[STAGE1] Error accepting Stage 2 connection\n");
        socket_close(forward_server);
        cleanup_batch_tracker(&batch_tracker);
        cleanup_buffer(&pipeline_buffer);
        free_stage(stage1);
        imgs_free(imgs, number_imgs);
        return;
    }
    
    // Connect to Stage 2 for backward pass with retry
    printf("[STAGE1] Attempting to connect to Stage 2 backward server...\n");
    fflush(stdout);
    
    int backward_client = -1;
    for (int retry = 0; retry < 10; retry++) {
        sleep(1); // Give Stage 2 time to set up backward server
        backward_client = connect_to_server("172.32.0.3", backward_port);
        if (backward_client >= 0) {
            printf("[STAGE1] Successfully connected to Stage 2 backward server\n");
            fflush(stdout);
            break;
        }
        
        printf("[STAGE1] Backward connection attempt %d failed, retrying...\n", retry + 1);
        fflush(stdout);
    }
    
    if (backward_client < 0) {
        printf("[STAGE1] Error connecting to Stage 2 backward server after retries\n");
        socket_close(forward_client);
        socket_close(forward_server);
        cleanup_batch_tracker(&batch_tracker);
        cleanup_buffer(&pipeline_buffer);
        free_stage(stage1);
        imgs_free(imgs, number_imgs);
        return;
    }
    
    printf("[STAGE1] Connected to Stage 2. Starting optimized pipeline training...\n");
    fflush(stdout);
    
    int current_batch_id = 0;
    int epochs = 5;
    
    gettimeofday(&start_time, NULL);
    
    for (int epoch = 0; epoch < epochs && training_active; epoch++) {
        printf("[STAGE1] Starting epoch %d/%d with adaptive batch size: %d\n", 
               epoch + 1, epochs, config.current_batch_size);
        fflush(stdout);
        
        for (int i = 0; i < number_imgs && training_active; i += config.current_batch_size) {
            // Phase 3: Dynamic batch size adjustment
            if (i > 0 && (i / config.current_batch_size) % 10 == 0) {
                PipelineStats current_stats = collect_pipeline_stats();
                update_pipeline_config(&config, &current_stats);
                int new_batch_size = calculate_adaptive_batch_size(&current_stats, &config);
                if (new_batch_size != config.current_batch_size) {
                    printf("[STAGE1] Adjusting batch size from %d to %d (comm_ratio: %.3f)\n", 
                           config.current_batch_size, new_batch_size, config.communication_ratio);
                    config.current_batch_size = new_batch_size;
                }
            }
            
            int batch_size = (i + config.current_batch_size > number_imgs) ? 
                           (number_imgs - i) : config.current_batch_size;
            
            // Prepare mini-batch
            Img** mini_batch = &imgs[i];
            int* labels = (int*)malloc(sizeof(int) * batch_size);
            for (int j = 0; j < batch_size; j++) {
                labels[j] = mini_batch[j]->label;
            }
        
            struct timeval stage1_start, stage1_end;
            gettimeofday(&stage1_start, NULL);
            
            // Create input matrix for later use in backprop
            Matrix* input_matrix = matrix_create(784, batch_size);
            for (int img_idx = 0; img_idx < batch_size; img_idx++) {
                Img* img = mini_batch[img_idx];
                for (int pixel_row = 0; pixel_row < 28; pixel_row++) {
                    for (int pixel_col = 0; pixel_col < 28; pixel_col++) {
                        int pixel_idx = pixel_row * 28 + pixel_col;
                        input_matrix->entries[pixel_idx][img_idx] = img->img_data->entries[pixel_row][pixel_col];
                    }
                }
            }
            
            // Forward pass through stage 1
            Matrix* hidden_activations = stage1_forward(stage1, mini_batch, batch_size);
            if (!hidden_activations) {
                printf("[STAGE1] Error in forward pass\n");
                matrix_free(input_matrix);
                free(labels);
                continue;
            }
            
            gettimeofday(&stage1_end, NULL);
            double stage1_time = get_time_diff(stage1_start, stage1_end);
            
            // Phase 2: Add to pipeline tracker for asynchronous processing
            Matrix* saved_activations = matrix_create(hidden_activations->rows, hidden_activations->cols);
            Matrix* saved_inputs = matrix_create(input_matrix->rows, input_matrix->cols);
            
            // Copy matrices for pipeline tracking
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
            
            // Create and send forward message
            ForwardMessage* fwd_msg = create_forward_message(
                current_batch_id, i / config.current_batch_size, 
                hidden_activations, labels, batch_size
            );
        
            if (fwd_msg) {
                struct timeval comm_start, comm_end;
                gettimeofday(&comm_start, NULL);
                
                if (send_forward_activations(forward_client, fwd_msg) < 0) {
                    printf("[STAGE1] Error sending forward message\n");
                    free_forward_message(fwd_msg);
                    matrix_free(hidden_activations);
                    matrix_free(input_matrix);
                    matrix_free(saved_activations);
                    matrix_free(saved_inputs);
                    free(labels);
                    continue;
                }
                
                gettimeofday(&comm_end, NULL);
                double comm_time = get_time_diff(comm_start, comm_end);
                
                // Add batch to pipeline tracker for asynchronous processing (ONLY ONCE)
                add_pending_batch(&batch_tracker, current_batch_id, saved_activations, saved_inputs);
                current_batch_id++;
                
                // Update stats immediately (without waiting for gradients)
                pthread_mutex_lock(&stats_mutex);
                global_stats.stage1_processing_time += stage1_time;
                global_stats.communication_time += comm_time;
                global_stats.processed_batches++;
                pthread_mutex_unlock(&stats_mutex);
                
                // Process any available gradients asynchronously (but don't block!)
                int processed_gradients = 0;
                printf("[STAGE1] Processing async gradients, pending: %d\n", batch_tracker.pending_count);
                
                while (processed_gradients < 2) { // Process up to 2 gradients per batch send
                    BackwardMessage* bwd_msg = receive_backward_gradients_timeout(backward_client, 10); // 10ms timeout
                    if (!bwd_msg) {
                        printf("[STAGE1] No gradients available after 10ms timeout\n");
                        break; // No gradients available
                    }
                    
                    printf("[STAGE1] Received gradient for batch_id %d\n", bwd_msg->batch_id);
                    
                    if (apply_gradient_to_pending_batch(stage1, &batch_tracker, bwd_msg) == 0) {
                        processed_gradients++;
                        
                        // Update loss/accuracy stats from gradient
                        pthread_mutex_lock(&stats_mutex);
                        global_stats.total_loss += bwd_msg->loss;
                        global_stats.correct_predictions += bwd_msg->correct_predictions;
                        global_stats.total_predictions += bwd_msg->total_predictions;
                        pthread_mutex_unlock(&stats_mutex);
                        
                        printf("[STAGE1] Applied gradient for batch_id %d, loss: %.4f, acc: %d/%d\n", 
                               bwd_msg->batch_id, bwd_msg->loss, bwd_msg->correct_predictions, bwd_msg->total_predictions);
                    } else {
                        printf("[STAGE1] Failed to apply gradient for batch_id %d\n", bwd_msg->batch_id);
                    }
                    
                    free_backward_message(bwd_msg);
                }
                
                printf("[STAGE1] Processed %d gradients, pending: %d\n", processed_gradients, batch_tracker.pending_count);
                
                // Only enforce pipeline depth BEFORE next epoch or when really necessary
                if (batch_tracker.pending_count > config.pipeline_depth * 2) {
                    printf("[STAGE1] Pipeline overflow (%d), waiting for gradients...\n", batch_tracker.pending_count);
                    while (batch_tracker.pending_count > config.pipeline_depth) {
                        BackwardMessage* bwd_msg = receive_backward_gradients_timeout(backward_client, 100); // 100ms timeout
                        if (bwd_msg) {
                            apply_gradient_to_pending_batch(stage1, &batch_tracker, bwd_msg);
                            free_backward_message(bwd_msg);
                        } else {
                            break; // Timeout, continue
                        }
                    }
                }
                
                // Periodic logging and stats reporting
                if ((i / config.current_batch_size) % 50 == 0) {
                    printf("[STAGE1] ASYNC: Sent batch %d, processed %d gradients, pending: %d, pipeline_depth: %d\n", 
                           i / config.current_batch_size, processed_gradients, batch_tracker.pending_count, config.pipeline_depth);
                    fflush(stdout);
                    
                    // Send stats to coordinator
                    int stats_client = connect_stats_client("172.32.0.4", 12347);
                    if (stats_client >= 0) {
                        StatsMessage stats_msg = {
                            .stage_id = 1,
                            .processing_time = stage1_time,
                            .communication_time = comm_time,
                            .loss = 0.0, // Will be updated by gradients
                            .batch_count = 1,
                            .correct_predictions = 0, // Will be updated by gradients
                            .total_predictions = batch_size,
                            .timestamp = time(NULL)
                        };
                        send_stats_message(stats_client, &stats_msg);
                        socket_close(stats_client);
                    }
                }
                
                free_forward_message(fwd_msg);
            }
        
            // DO NOT free matrices here - they are now owned by pipeline tracker
            // saved_activations and saved_inputs will be freed by apply_gradient_to_pending_batch
            matrix_free(hidden_activations); // Only free the original, not the saved copies
            matrix_free(input_matrix);       // Only free the original, not the saved copies  
            free(labels);
        }
        
        // Process remaining gradients at end of epoch
        printf("[STAGE1] Processing remaining %d gradients at end of epoch %d...\n", 
               batch_tracker.pending_count, epoch + 1);
        
        while (batch_tracker.pending_count > 0) {
            BackwardMessage* bwd_msg = receive_backward_gradients_timeout(backward_client, 1000); // 1 second timeout
            if (bwd_msg) {
                apply_gradient_to_pending_batch(stage1, &batch_tracker, bwd_msg);
                free_backward_message(bwd_msg);
                printf("[STAGE1] Processed cleanup gradient, pending: %d\n", batch_tracker.pending_count);
            } else {
                // Timeout - break to avoid infinite loop
                printf("[STAGE1] Cleanup timeout, %d batches remaining\n", batch_tracker.pending_count);
                break;
            }
        }
        
        printf("[STAGE1] Completed epoch %d with adaptive pipeline\n", epoch + 1);
        fflush(stdout);
    }
    
    gettimeofday(&end_time, NULL);
    double total_time = get_time_diff(start_time, end_time);
    printf("[STAGE1] Training completed in %.2f seconds\n", total_time);
    
    // Signal coordinator that training is done
    training_active = 0;
    
    // Send final termination signal to coordinator
    int stats_client = connect_stats_client("172.32.0.4", 12347);
    if (stats_client >= 0) {
        StatsMessage final_msg = {
            .stage_id = -1, // Special termination signal
            .processing_time = total_time,
            .communication_time = 0.0,
            .loss = 0.0,
            .batch_count = 0,
            .timestamp = time(NULL)
        };
        send_stats_message(stats_client, &final_msg);
        socket_close(stats_client);
    }
    
    printf("[STAGE1] Training completed. Starting evaluation on test set...\n");
    fflush(stdout);
    
    // Load test dataset for evaluation
    int test_number_imgs = 10000; // MNIST test set
    Img** test_imgs = csv_to_imgs("./data/mnist_test.csv", test_number_imgs);
    if (!test_imgs) {
        printf("[STAGE1] Error loading test dataset for evaluation\n");
    } else {
        printf("[STAGE1] Loaded %d test images for evaluation\n", test_number_imgs);
        fflush(stdout);
        
        int eval_correct = 0;
        int eval_total = 0;
        double eval_total_loss = 0.0;
        int eval_batches = 0;
        
        // Process test data in mini-batches (evaluation mode - no weight updates)
        for (int i = 0; i < test_number_imgs; i += config.current_batch_size) {
            int batch_size = (i + config.current_batch_size > test_number_imgs) ? 
                           (test_number_imgs - i) : config.current_batch_size;
            
            // Prepare mini-batch
            Img** mini_batch = &test_imgs[i];
            int* labels = (int*)malloc(sizeof(int) * batch_size);
            for (int j = 0; j < batch_size; j++) {
                labels[j] = mini_batch[j]->label;
            }
            
            // Forward pass through stage 1 (using trained weights)
            Matrix* hidden_activations = stage1_forward(stage1, mini_batch, batch_size);
            if (!hidden_activations) {
                printf("[STAGE1] Error in evaluation forward pass\n");
                free(labels);
                continue;
            }
            
            // Create and send forward message for evaluation
            ForwardMessage* fwd_msg = create_forward_message(
                -1, i / config.current_batch_size, // Use -1 to indicate evaluation mode
                hidden_activations, labels, batch_size
            );
            
            if (fwd_msg) {
                if (send_forward_activations(forward_client, fwd_msg) < 0) {
                    printf("[STAGE1] Error sending evaluation message\n");
                    free_forward_message(fwd_msg);
                    matrix_free(hidden_activations);
                    free(labels);
                    continue;
                }
                
                // Receive evaluation results from stage 2
                BackwardMessage* eval_msg = receive_backward_gradients(backward_client);
                if (eval_msg) {
                    eval_correct += eval_msg->correct_predictions;
                    eval_total += eval_msg->total_predictions;
                    eval_total_loss += eval_msg->loss;
                    eval_batches++;
                    
                    free_backward_message(eval_msg);
                }
                
                free_forward_message(fwd_msg);
            }
            
            matrix_free(hidden_activations);
            free(labels);
        }
        
        // Calculate and display final evaluation results
        double final_accuracy = (eval_total > 0) ? (double)eval_correct / eval_total * 100.0 : 0.0;
        double avg_eval_loss = (eval_batches > 0) ? eval_total_loss / eval_batches : 0.0;
        
        printf("\n");
        printf("================== FINAL EVALUATION RESULTS ==================\n");
        printf("Test Dataset: ./data/mnist_test.csv\n");
        printf("Total Test Samples: %d\n", eval_total);
        printf("Correct Predictions: %d\n", eval_correct);
        printf("Final Test Accuracy: %.2f%%\n", final_accuracy);
        printf("Average Test Loss: %.4f\n", avg_eval_loss);
        printf("==============================================================\n");
        fflush(stdout);
        
        imgs_free(test_imgs, test_number_imgs);
    }
    
    // Cleanup
    socket_close(forward_client);
    socket_close(backward_client);
    socket_close(forward_server);
    cleanup_batch_tracker(&batch_tracker);
    cleanup_buffer(&pipeline_buffer);
    free_stage(stage1);
    imgs_free(imgs, number_imgs);
}

void pipeline_stage2_main(int forward_port, int backward_port, const char* stage1_ip) {
    printf("[STAGE2] Starting Stage 2, connecting to %s on ports %d/%d\n", stage1_ip, forward_port, backward_port);
    fflush(stdout);
    
    // Create stage 2 network (hidden -> output) with proper learning rate
    NetworkStage* stage2 = create_stage(2, 512, 10, 0.01);
    if (!stage2) {
        printf("[STAGE2] Error creating network stage\n");
        return;
    }
    
    // Connect to Stage 1 for forward pass with retry
    printf("[STAGE2] Attempting to connect to Stage 1...\n");
    fflush(stdout);
    
    int forward_client = -1;
    for (int retry = 0; retry < 15; retry++) {
        forward_client = connect_to_server(stage1_ip, forward_port);
        if (forward_client >= 0) {
            printf("[STAGE2] Successfully connected to Stage 1 on attempt %d\n", retry + 1);
            fflush(stdout);
            break;
        }
        
        printf("[STAGE2] Connection attempt %d failed, retrying in 2s...\n", retry + 1);
        fflush(stdout);
        sleep(2);
    }
    
    if (forward_client < 0) {
        printf("[STAGE2] Error connecting to Stage 1 forward server after %d retries\n", 15);
        free_stage(stage2);
        return;
    }
    
    // Setup backward server
    int backward_server = setup_server(backward_port);
    if (backward_server < 0) {
        printf("[STAGE2] Error setting up backward server\n");
        socket_close(forward_client);
        free_stage(stage2);
        return;
    }
    
    int backward_client = accept_client(backward_server);
    if (backward_client < 0) {
        printf("[STAGE2] Error accepting Stage 1 backward connection\n");
        socket_close(forward_client);
        socket_close(backward_server);
        free_stage(stage2);
        return;
    }
    
    printf("[STAGE2] Connected to Stage 1. Ready for training...\n");
    fflush(stdout);
    
    while (training_active) {
        // Receive activations from stage 1
        ForwardMessage* fwd_msg = receive_forward_activations(forward_client);
        if (!fwd_msg) {
            printf("[STAGE2] Connection closed or error receiving forward message\n");
            break;
        }
        
        struct timeval stage2_start, stage2_end;
        gettimeofday(&stage2_start, NULL);
        
        // Convert received activations back to matrix
        Matrix* activations = matrix_create(512, fwd_msg->label_count); // 512 hidden units
        int idx = 0;
        for (int i = 0; i < activations->rows && idx < fwd_msg->activation_count; i++) {
            for (int j = 0; j < activations->cols && idx < fwd_msg->activation_count; j++) {
                activations->entries[i][j] = fwd_msg->activations[idx++];
            }
        }
        
        // Forward pass through stage 2
        Matrix* predictions = stage2_forward(stage2, activations);
        if (!predictions) {
            printf("[STAGE2] Error in forward pass\n");
            if (activations) matrix_free(activations);
            if (fwd_msg) free_forward_message(fwd_msg);
            continue;
        }
        
        // Calculate loss and accuracy
        double loss = calculate_pipeline_loss(predictions, fwd_msg->labels, fwd_msg->label_count);
        int correct = calculate_pipeline_accuracy(predictions, fwd_msg->labels, fwd_msg->label_count);
        
        // Backward pass through stage 2 with activations for proper weight updates
        int is_evaluation_mode = (fwd_msg->batch_id == -1);
        Matrix* gradients = stage2_backward(stage2, predictions, fwd_msg->labels, fwd_msg->label_count, activations, is_evaluation_mode);
        
        gettimeofday(&stage2_end, NULL);
        double stage2_time = get_time_diff(stage2_start, stage2_end);
        
        if (is_evaluation_mode) {
            // Evaluation mode: only send results, no gradients
            Matrix* dummy_gradients = matrix_create(1, 1);
            if (dummy_gradients) {
                dummy_gradients->entries[0][0] = 0.0; // dummy value
            }
            
            BackwardMessage* eval_msg = create_backward_message(
                fwd_msg->batch_id, fwd_msg->mini_batch_id, 
                dummy_gradients, loss, correct, fwd_msg->label_count
            );
            
            if (eval_msg) {
                if (send_backward_gradients(backward_client, eval_msg) < 0) {
                    printf("[STAGE2] Error sending evaluation results\n");
                }
                
                if (fwd_msg->mini_batch_id % 100 == 0) {
                    printf("[STAGE2] EVAL batch %d, loss: %.4f\n", 
                           fwd_msg->mini_batch_id, loss);
                    fflush(stdout);
                }
                
                free_backward_message(eval_msg);
            }
            
            if (dummy_gradients) matrix_free(dummy_gradients);
        } else {
            // Training mode: normal gradient computation and weight updates
            if (gradients) {
                BackwardMessage* bwd_msg = create_backward_message(
                    fwd_msg->batch_id, fwd_msg->mini_batch_id, gradients, loss, correct, fwd_msg->label_count
                );
                
                if (bwd_msg) {
                    if (send_backward_gradients(backward_client, bwd_msg) < 0) {
                        printf("[STAGE2] Error sending backward message\n");
                    }
                    
                    // Update global stats for stage2
                    pthread_mutex_lock(&stats_mutex);
                    global_stats.stage2_processing_time += stage2_time;
                    pthread_mutex_unlock(&stats_mutex);
                    
                    // Send stats to coordinator
                    if (fwd_msg->mini_batch_id % 50 == 0) { // Send stats every 50 batches
                        int stats_client = connect_stats_client("172.32.0.4", 12347);
                                                if (stats_client >= 0) {
                                StatsMessage stats_msg = {
                                    .stage_id = 2,
                                    .processing_time = stage2_time,
                                    .communication_time = 0.0,
                                    .loss = loss,
                                    .batch_count = 1,
                                    .correct_predictions = correct,
                                    .total_predictions = fwd_msg->label_count,
                                    .timestamp = time(NULL)
                                };
                                send_stats_message(stats_client, &stats_msg);
                                socket_close(stats_client);
                            }
                    }
                    
                    if (fwd_msg->mini_batch_id % 100 == 0) {
                        printf("[STAGE2] Processed batch %d, loss: %.4f, stage2_time: %.3fs\n", 
                               fwd_msg->mini_batch_id, loss, stage2_time);
                        fflush(stdout);
                    }
                    
                    if (bwd_msg) free_backward_message(bwd_msg);
                }
                
                // Always free gradients matrix as it's only copied to msg, not transferred
                matrix_free(gradients);
            }
        }
        
        // Update stage 2 weights using the activations from stage 1
        // This is a simplified update - in practice, we'd need more sophisticated gradient calculation
        
        // Free matrices in correct order to avoid double free
        if (activations) matrix_free(activations);
        if (predictions) matrix_free(predictions);
        if (fwd_msg) free_forward_message(fwd_msg);
    }
    
    printf("[STAGE2] Training completed\n");
    
    // Cleanup
    socket_close(forward_client);
    socket_close(backward_client);
    socket_close(backward_server);
    free_stage(stage2);
}

void coordinator_main(int stage1_port, int stage2_port) {
    printf("[COORDINATOR] Starting coordinator, monitoring ports %d/%d\n", stage1_port, stage2_port);
    fflush(stdout);
    
    // Setup stats server
    int stats_port = 12347;
    int stats_server = setup_server(stats_port);
    if (stats_server < 0) {
        printf("[COORDINATOR] Error setting up stats server\n");
        return;
    }
    
    printf("[COORDINATOR] Stats server listening on port %d\n", stats_port);
    fflush(stdout);
    
    // Aggregated stats
    PipelineStats aggregated_stats = {0};
    time_t start_time = time(NULL);
    int last_processed_batches = 0;
    
    while (training_active) {
        // Accept stats connections (non-blocking)
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        
        // Set non-blocking mode for stats collection
        fd_set read_fds;
        struct timeval timeout;
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;
        
        FD_ZERO(&read_fds);
        FD_SET(stats_server, &read_fds);
        
        int activity = select(stats_server + 1, &read_fds, NULL, NULL, &timeout);
        
        if (activity > 0 && FD_ISSET(stats_server, &read_fds)) {
            int stats_client = accept(stats_server, (struct sockaddr*)&client_addr, &client_len);
            if (stats_client >= 0) {
                StatsMessage* stats_msg = receive_stats_message(stats_client);
                if (stats_msg) {
                    // Check for termination signal
                    if (stats_msg->stage_id == -1) {
                        printf("[COORDINATOR] Received termination signal. Training completed.\n");
                        fflush(stdout);
                        free(stats_msg);
                        training_active = 0;
                        break;
                    }
                    
                    // Update aggregated stats
                    if (stats_msg->stage_id == 1) {
                        aggregated_stats.stage1_processing_time += stats_msg->processing_time;
                        aggregated_stats.communication_time += stats_msg->communication_time;
                        aggregated_stats.total_loss += stats_msg->loss;
                        aggregated_stats.correct_predictions += stats_msg->correct_predictions;
                        aggregated_stats.total_predictions += stats_msg->total_predictions;
                        aggregated_stats.processed_batches += stats_msg->batch_count;
                    } else if (stats_msg->stage_id == 2) {
                        aggregated_stats.stage2_processing_time += stats_msg->processing_time;
                    }
                    
                    free(stats_msg);
                }
                socket_close(stats_client);
            }
        }
        
        // Report stats every 5 seconds
        time_t current_time = time(NULL);
        double elapsed = difftime(current_time, start_time);
        
        if ((int)elapsed % 5 == 0 && aggregated_stats.processed_batches != last_processed_batches) {
            double avg_loss = (aggregated_stats.processed_batches > 0) ? 
                            aggregated_stats.total_loss / aggregated_stats.processed_batches : 0.0;
            
            double avg_accuracy = (aggregated_stats.total_predictions > 0) ?
                                (double)aggregated_stats.correct_predictions / aggregated_stats.total_predictions * 100.0 : 0.0;
            
            double throughput = (elapsed > 0) ? aggregated_stats.processed_batches / elapsed : 0.0;
            
            printf("[COORDINATOR] Runtime: %.0fs, Processed: %d batches, Avg loss: %.4f, Avg accuracy: %.2f%%, Throughput: %.2f batches/s\n", 
                   elapsed, aggregated_stats.processed_batches, avg_loss, avg_accuracy, throughput);
            printf("[COORDINATOR] Stage1 time: %.2fs, Stage2 time: %.2fs, Comm time: %.2fs\n",
                   aggregated_stats.stage1_processing_time, 
                   aggregated_stats.stage2_processing_time,
                   aggregated_stats.communication_time);
            fflush(stdout);
            
            last_processed_batches = aggregated_stats.processed_batches;
        }
        
        // Simple termination condition
        // if (elapsed > 300) { // 5 minutes max
        //     printf("[COORDINATOR] Training time limit reached\n");
        //     training_active = 0;
        //     break;
        // }
    }
    
    socket_close(stats_server);
    printf("[COORDINATOR] Monitoring completed\n");
}

