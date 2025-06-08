#include "pipeline_nn.h"
#include "../neural/activations.h"
#include "../matrix/ops.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>

// Global variables for stage management
extern PipelineStats global_stats;
extern pthread_mutex_t stats_mutex;
static int random_seeded = 0;

NetworkStage* create_stage(int stage_id, int input_size, int output_size, double lr) {
    NetworkStage* stage = (NetworkStage*)malloc(sizeof(NetworkStage));
    if (!stage) return NULL;
    
    // Seed random number generator once
    if (!random_seeded) {
        srand(time(NULL));
        random_seeded = 1;
        printf("[DEBUG] Random seed initialized\n");
    }
    
    stage->stage_id = stage_id;
    stage->input_size = input_size;
    stage->output_size = output_size;
    stage->learning_rate = lr;
    
    // Initialize weights and biases
    stage->weights = matrix_create(output_size, input_size);
    stage->biases = matrix_create(output_size, 1);
    
    if (!stage->weights || !stage->biases) {
        free_stage(stage);
        return NULL;
    }
    
    // Xavier/Glorot initialization: weights ~ N(0, sqrt(2/(input_size + output_size)))
    double xavier_std = sqrt(2.0 / (input_size + output_size));
    
    printf("[DEBUG] Stage %d: Xavier std = %.6f (input=%d, output=%d, lr=%.4f)\n", 
           stage_id, xavier_std, input_size, output_size, lr);
    
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size; j++) {
            // Box-Muller transform for normal distribution
            static int has_spare = 0;
            static double spare;
            
            if (has_spare) {
                stage->weights->entries[i][j] = spare * xavier_std;
                has_spare = 0;
            } else {
                double u = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
                double v = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
                double s = u * u + v * v;
                
                if (s >= 1.0 || s == 0.0) {
                    // Fallback to simple uniform if Box-Muller fails
                    stage->weights->entries[i][j] = (((double)rand() / RAND_MAX) * 2.0 - 1.0) * xavier_std;
                } else {
                    s = sqrt(-2.0 * log(s) / s);
                    stage->weights->entries[i][j] = u * s * xavier_std;
                    spare = v * s;
                    has_spare = 1;
                }
            }
        }
        stage->biases->entries[i][0] = 0.0; // Initialize biases to 0
    }
    
    return stage;
}

Matrix* stage1_forward(NetworkStage* stage, Img** imgs, int count) {
    if (!stage || !imgs || count <= 0) return NULL;
    
    // Create input matrix from images
    Matrix* input = matrix_create(784, count); // 28x28 = 784 pixels per image
    if (!input) return NULL;
    
    // Fill input matrix with image data
    for (int img_idx = 0; img_idx < count; img_idx++) {
        Img* img = imgs[img_idx];
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                int pixel_idx = i * 28 + j;
                input->entries[pixel_idx][img_idx] = img->img_data->entries[i][j] / 255.0; // Normalize
            }
        }
    }
    
    // Forward pass: z = W * x + b
    Matrix* z = dot(stage->weights, input);
    
    // Add bias to each column (image)
    for (int i = 0; i < z->rows; i++) {
        for (int j = 0; j < z->cols; j++) {
            z->entries[i][j] += stage->biases->entries[i][0];
        }
    }
    
    // Apply activation function (sigmoid)
    Matrix* activations = apply(sigmoid, z);
    
    // Clean up
    matrix_free(input);
    matrix_free(z);
    
    return activations;
}

Matrix* stage2_forward(NetworkStage* stage, Matrix* activations) {
    if (!stage || !activations) return NULL;
    
    // Forward pass: z = W * activations + b
    Matrix* z = dot(stage->weights, activations);
    
    // Add bias
    for (int i = 0; i < z->rows; i++) {
        for (int j = 0; j < z->cols; j++) {
            z->entries[i][j] += stage->biases->entries[i][0];
        }
    }
    
    // Apply ONLY softmax for final output (no sigmoid before!)
    Matrix* output = matrix_create(z->rows, z->cols);
    for (int i = 0; i < z->rows; i++) {
        for (int j = 0; j < z->cols; j++) {
            output->entries[i][j] = z->entries[i][j];
        }
    }
    
    // Apply softmax to each column (sample) with numerical stability
    for (int col = 0; col < output->cols; col++) {
        double sum = 0.0;
        
        // Find max for numerical stability
        double max_val = output->entries[0][col];
        for (int row = 1; row < output->rows; row++) {
            if (output->entries[row][col] > max_val) {
                max_val = output->entries[row][col];
            }
        }
        
        // Calculate exp sum with numerical stability
        for (int row = 0; row < output->rows; row++) {
            output->entries[row][col] = exp(output->entries[row][col] - max_val);
            sum += output->entries[row][col];
        }
        
        // Normalize to get probabilities
        for (int row = 0; row < output->rows; row++) {
            output->entries[row][col] /= sum;
        }
    }
    
    matrix_free(z);
    return output;
}

Matrix* stage2_backward(NetworkStage* stage, Matrix* predictions, int* labels, int count, Matrix* stage1_activations, int is_evaluation) {
    if (!stage || !predictions || !labels || count <= 0) return NULL;
    
    // Create target matrix
    Matrix* targets = matrix_create(predictions->rows, predictions->cols);
    for (int i = 0; i < count; i++) {
        for (int j = 0; j < predictions->rows; j++) {
            targets->entries[j][i] = (j == labels[i]) ? 1.0 : 0.0;
        }
    }
    
    // Calculate output errors: error = predictions - targets (for softmax + cross-entropy)
    Matrix* output_errors = matrix_create(predictions->rows, predictions->cols);
    for (int i = 0; i < predictions->rows; i++) {
        for (int j = 0; j < predictions->cols; j++) {
            output_errors->entries[i][j] = predictions->entries[i][j] - targets->entries[i][j];
        }
    }
    
    // Update stage 2 weights and biases if we have stage1 activations AND not in evaluation mode
    if (stage1_activations && !is_evaluation) {
        // Calculate weight gradients: dW = output_errors * stage1_activations^T / batch_size
        Matrix* stage1_activations_T = transpose(stage1_activations);
        Matrix* weight_gradients = dot(output_errors, stage1_activations_T);
        
        // Update weights: W = W - learning_rate * dW
        for (int i = 0; i < stage->weights->rows; i++) {
            for (int j = 0; j < stage->weights->cols; j++) {
                stage->weights->entries[i][j] -= stage->learning_rate * weight_gradients->entries[i][j] / count;
            }
        }
        
        // Update biases: db = mean(output_errors)
        for (int i = 0; i < stage->biases->rows; i++) {
            double bias_gradient = 0.0;
            for (int j = 0; j < output_errors->cols; j++) {
                bias_gradient += output_errors->entries[i][j];
            }
            stage->biases->entries[i][0] -= stage->learning_rate * bias_gradient / count;
        }
        
        matrix_free(stage1_activations_T);
        matrix_free(weight_gradients);
    }
    
    // Calculate gradients for stage 1: dE/dActivations = W^T * output_errors
    Matrix* stage_weights_T = transpose(stage->weights);
    Matrix* gradients_to_stage1 = dot(stage_weights_T, output_errors);
    
    // Clean up
    matrix_free(targets);
    matrix_free(output_errors);
    matrix_free(stage_weights_T);
    
    return gradients_to_stage1;
}

Matrix* stage1_backward(NetworkStage* stage, Matrix* gradients, Matrix* stage1_activations, Matrix* stage1_inputs, int is_evaluation) {
    if (!stage || !gradients) return NULL;
    
    // Apply derivative of sigmoid activation function
    // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    Matrix* sigmoid_derivative = matrix_create(gradients->rows, gradients->cols);
    for (int i = 0; i < gradients->rows; i++) {
        for (int j = 0; j < gradients->cols; j++) {
            if (stage1_activations) {
                double activation = stage1_activations->entries[i][j];
                sigmoid_derivative->entries[i][j] = activation * (1.0 - activation);
            } else {
                // Fallback if activations not available
                sigmoid_derivative->entries[i][j] = 0.25; // Approximate maximum of sigmoid derivative
            }
        }
    }
    
    // Apply chain rule: local_gradients = gradients * sigmoid_derivative
    Matrix* local_gradients = matrix_create(gradients->rows, gradients->cols);
    for (int i = 0; i < gradients->rows; i++) {
        for (int j = 0; j < gradients->cols; j++) {
            local_gradients->entries[i][j] = gradients->entries[i][j] * sigmoid_derivative->entries[i][j];
        }
    }
    
    // Update stage 1 weights and biases if we have inputs AND not in evaluation mode
    if (stage1_inputs && !is_evaluation) {
        // Calculate weight gradients: dW = local_gradients * inputs^T / batch_size
        Matrix* inputs_T = transpose(stage1_inputs);
        Matrix* weight_gradients = dot(local_gradients, inputs_T);
        
        // Update weights
        for (int i = 0; i < stage->weights->rows; i++) {
            for (int j = 0; j < stage->weights->cols; j++) {
                stage->weights->entries[i][j] -= stage->learning_rate * weight_gradients->entries[i][j] / gradients->cols;
            }
        }
        
        // Update biases
        for (int i = 0; i < stage->biases->rows; i++) {
            double bias_gradient = 0.0;
            for (int j = 0; j < local_gradients->cols; j++) {
                bias_gradient += local_gradients->entries[i][j];
            }
            stage->biases->entries[i][0] -= stage->learning_rate * bias_gradient / gradients->cols;
        }
        
        matrix_free(inputs_T);
        matrix_free(weight_gradients);
    }
    
    matrix_free(sigmoid_derivative);
    return local_gradients;
}

void update_stage_weights(NetworkStage* stage, Matrix* gradients) {
    if (!stage || !gradients) return;
    
    // This function is now deprecated in favor of weight updates in backward functions
    // Keep for compatibility but weights are updated in stage1_backward and stage2_backward
    printf("[WARNING] update_stage_weights is deprecated. Use backward functions for weight updates.\n");
}

void free_stage(NetworkStage* stage) {
    if (!stage) return;
    
    if (stage->weights) matrix_free(stage->weights);
    if (stage->biases) matrix_free(stage->biases);
    free(stage);
}

double calculate_pipeline_loss(Matrix* predictions, int* labels, int count) {
    if (!predictions || !labels || count <= 0) return 0.0;
    
    double total_loss = 0.0;
    
    // Calculate cross-entropy loss
    for (int i = 0; i < count; i++) {
        double loss = 0.0;
        for (int j = 0; j < predictions->rows; j++) {
            if (j == labels[i]) {
                // Avoid log(0) by adding small epsilon
                double pred = predictions->entries[j][i];
                if (pred < 1e-15) pred = 1e-15;
                if (pred > 1.0 - 1e-15) pred = 1.0 - 1e-15;
                loss -= log(pred);
            }
        }
        total_loss += loss;
    }
    
    return total_loss / count; // Average loss
}

int calculate_pipeline_accuracy(Matrix* predictions, int* labels, int count) {
    if (!predictions || !labels || count <= 0) return 0;
    
    int correct = 0;
    
    // Calculate accuracy by finding the maximum prediction for each sample
    for (int i = 0; i < count; i++) {
        int predicted_class = 0;
        double max_prob = predictions->entries[0][i];
        
        // Find the class with highest probability
        for (int j = 1; j < predictions->rows; j++) {
            if (predictions->entries[j][i] > max_prob) {
                max_prob = predictions->entries[j][i];
                predicted_class = j;
            }
        }
        
        // Check if prediction matches true label
        if (predicted_class == labels[i]) {
            correct++;
        }
    }
    
    return correct;
}

PipelineStats collect_pipeline_stats(void) {
    PipelineStats stats;
    pthread_mutex_lock(&stats_mutex);
    stats = global_stats;
    
    // Calculate throughput
    if (global_stats.stage1_processing_time > 0) {
        stats.throughput = (double)global_stats.processed_batches / global_stats.stage1_processing_time;
    } else {
        stats.throughput = 0.0;
    }
    
    pthread_mutex_unlock(&stats_mutex);
    return stats;
}

int calculate_optimal_batch_size(PipelineStats* stats) {
    if (!stats) return MINI_BATCH_SIZE;
    
    // Simple heuristic: if communication time is high, increase batch size
    // if processing time is high, decrease batch size
    
    double comm_ratio = stats->communication_time / (stats->stage1_processing_time + stats->stage2_processing_time + 1e-6);
    
    int optimal_size = MINI_BATCH_SIZE;
    
    if (comm_ratio > 0.5) {
        // Communication is bottleneck, increase batch size
        optimal_size = MINI_BATCH_SIZE * 2;
    } else if (comm_ratio < 0.1) {
        // Processing is bottleneck, decrease batch size
        optimal_size = MINI_BATCH_SIZE / 2;
    }
    
    // Keep within reasonable bounds
    if (optimal_size < 8) optimal_size = 8;
    if (optimal_size > 128) optimal_size = 128;
    
    return optimal_size;
}

// Phase 2: Batch tracking implementation
long get_current_timestamp(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

int initialize_batch_tracker(PipelineBatchTracker* tracker) {
    if (!tracker) return -1;
    
    tracker->pending_count = 0;
    tracker->next_expected_gradient_id = 0;
    tracker->buffered_count = 0;
    
    if (pthread_mutex_init(&tracker->tracker_mutex, NULL) != 0) {
        return -1;
    }
    
    for (int i = 0; i < MAX_PIPELINE_DEPTH; i++) {
        tracker->pending_batches[i].batch_id = -1;
        tracker->pending_batches[i].saved_activations = NULL;
        tracker->pending_batches[i].saved_inputs = NULL;
        tracker->buffered_gradients[i] = NULL;
    }
    
    return 0;
}

int add_pending_batch(PipelineBatchTracker* tracker, int batch_id, Matrix* activations, Matrix* inputs) {
    if (!tracker || tracker->pending_count >= MAX_PIPELINE_DEPTH) return -1;
    
    pthread_mutex_lock(&tracker->tracker_mutex);
    
    int slot = tracker->pending_count;
    tracker->pending_batches[slot].batch_id = batch_id;
    tracker->pending_batches[slot].saved_activations = activations;
    tracker->pending_batches[slot].saved_inputs = inputs;
    tracker->pending_batches[slot].timestamp = get_current_timestamp();
    tracker->pending_count++;
    
    pthread_mutex_unlock(&tracker->tracker_mutex);
    return 0;
}

int process_pending_gradients(PipelineBatchTracker* tracker, int backward_client) {
    if (!tracker) return -1;
    
    pthread_mutex_lock(&tracker->tracker_mutex);
    
    // Process any buffered gradients that match expected batch ID
    for (int i = 0; i < tracker->buffered_count; i++) {
        BackwardMessage* grad_msg = tracker->buffered_gradients[i];
        if (grad_msg && grad_msg->batch_id == tracker->next_expected_gradient_id) {
            // Find corresponding pending batch
            for (int j = 0; j < tracker->pending_count; j++) {
                if (tracker->pending_batches[j].batch_id == grad_msg->batch_id) {
                    // Apply gradients using saved context
                    // This would be implemented in the main processing loop
                    
                    // Clean up this batch
                    tracker->pending_batches[j].batch_id = -1;
                    if (tracker->pending_batches[j].saved_activations) {
                        matrix_free(tracker->pending_batches[j].saved_activations);
                        tracker->pending_batches[j].saved_activations = NULL;
                    }
                    if (tracker->pending_batches[j].saved_inputs) {
                        matrix_free(tracker->pending_batches[j].saved_inputs);
                        tracker->pending_batches[j].saved_inputs = NULL;
                    }
                    
                    tracker->next_expected_gradient_id++;
                    break;
                }
            }
            
            // Remove from buffered gradients
            free_backward_message(grad_msg);
            tracker->buffered_gradients[i] = NULL;
            
            // Compact the buffer
            for (int k = i; k < tracker->buffered_count - 1; k++) {
                tracker->buffered_gradients[k] = tracker->buffered_gradients[k + 1];
            }
            tracker->buffered_count--;
            i--; // Adjust index after compaction
        }
    }
    
    pthread_mutex_unlock(&tracker->tracker_mutex);
    return 0;
}

void cleanup_batch_tracker(PipelineBatchTracker* tracker) {
    if (!tracker) return;
    
    pthread_mutex_lock(&tracker->tracker_mutex);
    
    // Clean up pending batches
    for (int i = 0; i < tracker->pending_count; i++) {
        if (tracker->pending_batches[i].saved_activations) {
            matrix_free(tracker->pending_batches[i].saved_activations);
        }
        if (tracker->pending_batches[i].saved_inputs) {
            matrix_free(tracker->pending_batches[i].saved_inputs);
        }
    }
    
    // Clean up buffered gradients
    for (int i = 0; i < tracker->buffered_count; i++) {
        if (tracker->buffered_gradients[i]) {
            free_backward_message(tracker->buffered_gradients[i]);
        }
    }
    
    pthread_mutex_unlock(&tracker->tracker_mutex);
    pthread_mutex_destroy(&tracker->tracker_mutex);
}

// Phase 3: Dynamic load balancing implementation
void update_pipeline_config(PipelineConfig* config, PipelineStats* stats) {
    if (!config || !stats) return;
    
    long current_time = get_current_timestamp();
    
    // Only adjust every 5 seconds to avoid thrashing
    if (current_time - config->last_adjustment_time < 5000000) return;
    
    // Calculate communication ratio
    double total_processing = stats->stage1_processing_time + stats->stage2_processing_time;
    if (total_processing > 0) {
        config->communication_ratio = stats->communication_time / total_processing;
    }
    
    // Adjust batch size based on communication ratio
    if (config->communication_ratio > 0.3) {
        // Communication is bottleneck, increase batch size
        config->current_batch_size = min(config->current_batch_size * 1.2, 64);
    } else if (config->communication_ratio < 0.1) {
        // Processing is bottleneck, decrease batch size for better pipelining
        config->current_batch_size = max(config->current_batch_size * 0.8, 16);
    }
    
    config->last_adjustment_time = current_time;
}

int calculate_adaptive_batch_size(PipelineStats* stats, PipelineConfig* config) {
    if (!stats || !config) return MINI_BATCH_SIZE;
    
    // Base the batch size on throughput and latency
    double avg_processing_time = (stats->stage1_processing_time + stats->stage2_processing_time) / 
                                max(stats->processed_batches, 1);
    
    // If processing time per batch is too high, reduce batch size
    if (avg_processing_time > config->target_latency) {
        return max(config->current_batch_size - 4, 8);
    }
    // If processing time is low, we can afford larger batches
    else if (avg_processing_time < config->target_latency * 0.5) {
        return min(config->current_batch_size + 4, 128);
    }
    
    return config->current_batch_size;
}

void adjust_pipeline_depth(PipelineConfig* config, PipelineStats* stats) {
    if (!config || !stats) return;
    
    // Increase pipeline depth if we have high throughput but low utilization
    if (stats->throughput > 10.0 && config->communication_ratio < 0.2) {
        config->pipeline_depth = min(config->pipeline_depth + 1, MAX_PIPELINE_DEPTH);
    }
    // Decrease pipeline depth if communication is becoming a bottleneck
    else if (config->communication_ratio > 0.4) {
        config->pipeline_depth = max(config->pipeline_depth - 1, 2);
    }
}

// Phase 2: Asynchronous processing implementations
void* async_forward_processor(void* args) {
    AsyncForwardArgs* fwd_args = (AsyncForwardArgs*)args;
    if (!fwd_args) return NULL;
    
    printf("[ASYNC_FORWARD] Starting asynchronous forward processor\n");
    fflush(stdout);
    
    while (*fwd_args->training_active) {
        // Dequeue forward message from buffer
        ForwardMessage* fwd_msg = dequeue_forward(fwd_args->buffer);
        if (!fwd_msg) {
            usleep(1000); // 1ms sleep if no messages
            continue;
        }
        
        // Send forward activations to next stage
        struct timeval comm_start, comm_end;
        gettimeofday(&comm_start, NULL);
        
        if (send_forward_activations(fwd_args->forward_socket, fwd_msg) < 0) {
            printf("[ASYNC_FORWARD] Error sending forward message\n");
            free_forward_message(fwd_msg);
            continue;
        }
        
        gettimeofday(&comm_end, NULL);
        double comm_time = get_time_diff(comm_start, comm_end);
        
        // Update communication stats
        pthread_mutex_lock(&stats_mutex);
        global_stats.communication_time += comm_time;
        pthread_mutex_unlock(&stats_mutex);
        
        free_forward_message(fwd_msg);
    }
    
    printf("[ASYNC_FORWARD] Stopping asynchronous forward processor\n");
    return NULL;
}

void* async_backward_processor(void* args) {
    AsyncBackwardArgs* bwd_args = (AsyncBackwardArgs*)args;
    if (!bwd_args) return NULL;
    
    printf("[ASYNC_BACKWARD] Starting asynchronous backward processor\n");
    fflush(stdout);
    
    while (*bwd_args->training_active) {
        // Receive backward gradients
        BackwardMessage* bwd_msg = receive_backward_gradients(bwd_args->backward_socket);
        if (!bwd_msg) {
            usleep(1000); // 1ms sleep if no messages
            continue;
        }
        
        // Add to pipeline buffer for processing
        if (enqueue_backward(bwd_args->buffer, bwd_msg) < 0) {
            printf("[ASYNC_BACKWARD] Error enqueuing backward message\n");
            free_backward_message(bwd_msg);
            continue;
        }
        
        // Process pending gradients asynchronously
        process_pending_gradients(bwd_args->tracker, bwd_args->backward_socket);
    }
    
    printf("[ASYNC_BACKWARD] Stopping asynchronous backward processor\n");
    return NULL;
}

// Asynchronous gradient application function
int apply_gradient_to_pending_batch(NetworkStage* stage, PipelineBatchTracker* tracker, BackwardMessage* bwd_msg) {
    if (!stage || !tracker || !bwd_msg) return -1;
    
    pthread_mutex_lock(&tracker->tracker_mutex);
    
    // Find corresponding pending batch
    int found_batch = -1;
    for (int i = 0; i < tracker->pending_count; i++) {
        if (tracker->pending_batches[i].batch_id == bwd_msg->batch_id) {
            found_batch = i;
            break;
        }
    }
    
    if (found_batch == -1) {
        // Buffer this gradient for later if batch not found yet
        if (tracker->buffered_count < MAX_PIPELINE_DEPTH) {
            tracker->buffered_gradients[tracker->buffered_count] = malloc(sizeof(BackwardMessage));
            if (tracker->buffered_gradients[tracker->buffered_count]) {
                // Deep copy the backward message
                memcpy(tracker->buffered_gradients[tracker->buffered_count], bwd_msg, sizeof(BackwardMessage));
                
                // Deep copy gradients array
                tracker->buffered_gradients[tracker->buffered_count]->gradients = malloc(sizeof(double) * bwd_msg->gradient_count);
                if (tracker->buffered_gradients[tracker->buffered_count]->gradients) {
                    memcpy(tracker->buffered_gradients[tracker->buffered_count]->gradients, 
                           bwd_msg->gradients, sizeof(double) * bwd_msg->gradient_count);
                    tracker->buffered_count++;
                    printf("[ASYNC] Buffered gradient for batch_id %d (not yet sent)\n", bwd_msg->batch_id);
                }
            }
        }
        pthread_mutex_unlock(&tracker->tracker_mutex);
        return 0; // Not an error, just buffered
    }
    
    BatchContext* batch_ctx = &tracker->pending_batches[found_batch];
    
    // Apply gradient using saved context
    int batch_size = (batch_ctx->saved_activations) ? batch_ctx->saved_activations->cols : 32;
    
    // Convert gradients back to matrix form
    Matrix* gradients_from_stage2 = matrix_create(512, batch_size);
    if (gradients_from_stage2) {
        int idx = 0;
        for (int r = 0; r < 512 && idx < bwd_msg->gradient_count; r++) {
            for (int c = 0; c < batch_size && idx < bwd_msg->gradient_count; c++) {
                gradients_from_stage2->entries[r][c] = bwd_msg->gradients[idx++];
            }
        }
        
        // Perform backpropagation with saved context
        int is_eval_mode = (bwd_msg->batch_id == -1);
        Matrix* stage1_gradients = stage1_backward(stage, gradients_from_stage2, 
                                                  batch_ctx->saved_activations, 
                                                  batch_ctx->saved_inputs, is_eval_mode);
        
        // Clean up processed batch
        matrix_free(batch_ctx->saved_activations);
        matrix_free(batch_ctx->saved_inputs);
        batch_ctx->batch_id = -1;
        batch_ctx->saved_activations = NULL;
        batch_ctx->saved_inputs = NULL;
        
        // Compact pending batches array
        for (int j = found_batch; j < tracker->pending_count - 1; j++) {
            tracker->pending_batches[j] = tracker->pending_batches[j + 1];
        }
        tracker->pending_count--;
        
        matrix_free(gradients_from_stage2);
        if (stage1_gradients) matrix_free(stage1_gradients);
        
        printf("[ASYNC] Applied gradient for batch_id %d, pending: %d\n", bwd_msg->batch_id, tracker->pending_count);
    }
    
    pthread_mutex_unlock(&tracker->tracker_mutex);
    return 0;
}

 