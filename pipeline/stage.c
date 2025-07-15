/**
 * @file stage.c
 * @brief Neural network stage implementation for pipeline parallelism
 * @author Research Team
 * @date 2024
 * @version 1.0
 * 
 * This file implements the neural network processing stages for the pipeline
 * parallelism system. It contains the forward and backward pass implementations
 * for both Stage 1 (input to hidden layer) and Stage 2 (hidden to output layer).
 * 
 * Stage 1: Input layer (784) -> Hidden layer (512) with sigmoid activation
 * Stage 2: Hidden layer (512) -> Output layer (10) with softmax activation
 * 
 * Key features:
 * - Xavier/Glorot weight initialization for stable training
 * - Numerically stable softmax implementation
 * - Proper gradient computation for backpropagation
 * - Thread-safe statistics collection
 * - Comprehensive loss and accuracy calculation
 */

#define _DEFAULT_SOURCE
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

/* =============================================================================
 * GLOBAL VARIABLES FOR STAGE MANAGEMENT
 * ============================================================================= */

/** @brief External global statistics structure */
extern PipelineStats global_stats;

/** @brief External mutex for thread-safe statistics access */
extern pthread_mutex_t stats_mutex;

/** @brief External variables for metrics tracking */
extern int total_out_of_order_messages;
extern int current_epoch;

/** @brief Flag to ensure random number generator is seeded only once */
static int random_seeded = 0;

/**
 * @brief Create a neural network stage with Xavier/Glorot weight initialization
 * 
 * Creates and initializes a neural network stage with the specified architecture.
 * Uses Xavier/Glorot initialization for weights and zero initialization for biases.
 * This initialization method helps maintain stable gradients during training.
 * 
 * @param stage_id Unique identifier for this stage
 * @param input_size Number of input features
 * @param output_size Number of output features
 * @param lr Learning rate for gradient descent
 * @return Pointer to initialized NetworkStage, or NULL on failure
 */
NetworkStage* create_stage(int stage_id, int input_size, int output_size, double lr) {
    NetworkStage* stage = (NetworkStage*)malloc(sizeof(NetworkStage));
    if (!stage) return NULL;
    
    /* Initialize random number generator once */
    if (!random_seeded) {
        srand(time(NULL));
        random_seeded = 1;
    }
    
    stage->stage_id = stage_id;
    stage->input_size = input_size;
    stage->output_size = output_size;
    stage->learning_rate = lr;
    
    /* Initialize weight and bias matrices */
    stage->weights = matrix_create(output_size, input_size);
    stage->biases = matrix_create(output_size, 1);
    
    if (!stage->weights || !stage->biases) {
        free_stage(stage);
        return NULL;
    }
    
    /* Xavier/Glorot initialization: weights ~ N(0, sqrt(2/(input_size + output_size))) */
    double xavier_std = sqrt(2.0 / (input_size + output_size));
    
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size; j++) {
            /* Box-Muller transformation for normal distribution */
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
                    /* Fallback to simple uniform if Box-Muller fails */
                    stage->weights->entries[i][j] = (((double)rand() / RAND_MAX) * 2.0 - 1.0) * xavier_std;
                } else {
                    s = sqrt(-2.0 * log(s) / s);
                    stage->weights->entries[i][j] = u * s * xavier_std;
                    spare = v * s;
                    has_spare = 1;
                }
            }
        }
        stage->biases->entries[i][0] = 0.0; /* Initialize biases to zero */
    }
    
    return stage;
}

/**
 * @brief Stage 1 forward pass: Input layer to hidden layer transformation
 * 
 * Performs the forward pass for Stage 1, transforming input images (784 features)
 * to hidden layer activations (512 units) with sigmoid activation function.
 * 
 * @param stage Pointer to the Stage 1 network structure
 * @param imgs Array of input images (MNIST format)
 * @param count Number of images in the batch
 * @return Matrix of activations (512 x count), or NULL on failure
 */
Matrix* stage1_forward(NetworkStage* stage, Img** imgs, int count) {
    if (!stage || !imgs || count <= 0) return NULL;
    
    /* Create input matrix from images */
    Matrix* input = matrix_create(784, count); /* 28x28 = 784 pixels per image */
    if (!input) return NULL;
    
    /* Fill input matrix with image data */
    for (int img_idx = 0; img_idx < count; img_idx++) {
        Img* img = imgs[img_idx];
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                int pixel_idx = i * 28 + j;
                input->entries[pixel_idx][img_idx] = img->img_data->entries[i][j] / 255.0; /* Normalize to [0,1] */
            }
        }
    }
    
    /* Forward pass: z = W * x + b */
    Matrix* z = dot(stage->weights, input);
    
    /* Add bias for each column (image) */
    for (int i = 0; i < z->rows; i++) {
        for (int j = 0; j < z->cols; j++) {
            z->entries[i][j] += stage->biases->entries[i][0];
        }
    }
    
    /* Apply activation function (sigmoid) */
    Matrix* activations = apply(sigmoid, z);
    
    /* Cleanup intermediate matrices */
    matrix_free(input);
    matrix_free(z);
    
    return activations;
}

/**
 * @brief Stage 2 forward pass: Hidden layer to output layer with softmax
 * 
 * Performs the forward pass for Stage 2, transforming hidden layer activations
 * (512 units) to output probabilities (10 classes) using softmax activation.
 * Implements numerically stable softmax to prevent overflow.
 * 
 * @param stage Pointer to the Stage 2 network structure
 * @param activations Input activations from Stage 1 (512 x batch_size)
 * @return Matrix of output probabilities (10 x batch_size), or NULL on failure
 */
Matrix* stage2_forward(NetworkStage* stage, Matrix* activations) {
    if (!stage || !activations) return NULL;
    
    /* Forward pass: z = W * activations + b */
    Matrix* z = dot(stage->weights, activations);
    
    /* Add bias terms */
    for (int i = 0; i < z->rows; i++) {
        for (int j = 0; j < z->cols; j++) {
            z->entries[i][j] += stage->biases->entries[i][0];
        }
    }
    
    /* Apply softmax activation for final output (no sigmoid before!) */
    Matrix* output = matrix_create(z->rows, z->cols);
    for (int i = 0; i < z->rows; i++) {
        for (int j = 0; j < z->cols; j++) {
            output->entries[i][j] = z->entries[i][j];
        }
    }
    
    /* Apply numerically stable softmax for each column (sample) */
    for (int col = 0; col < output->cols; col++) {
        double sum = 0.0;
        
        /* Find maximum for numerical stability */
        double max_val = output->entries[0][col];
        for (int row = 1; row < output->rows; row++) {
            if (output->entries[row][col] > max_val) {
                max_val = output->entries[row][col];
            }
        }
        
        /* Compute exp sum with numerical stability */
        for (int row = 0; row < output->rows; row++) {
            output->entries[row][col] = exp(output->entries[row][col] - max_val);
            sum += output->entries[row][col];
        }
        
        /* Normalize to get probabilities */
        for (int row = 0; row < output->rows; row++) {
            output->entries[row][col] /= sum;
        }
    }
    
    matrix_free(z);
    return output;
}

/**
 * @brief Stage 2 backward pass: Compute gradients and update weights
 * 
 * Performs the backward pass for Stage 2, computing gradients for cross-entropy loss
 * with softmax activation. Updates Stage 2 weights and biases if not in evaluation mode.
 * 
 * @param stage Pointer to the Stage 2 network structure
 * @param predictions Output predictions from forward pass (10 x batch_size)
 * @param labels Ground truth labels array (batch_size)
 * @param count Number of samples in the batch
 * @param stage1_activations Activations from Stage 1 (512 x batch_size)
 * @param is_evaluation Flag indicating evaluation mode (no weight updates)
 * @return Matrix of gradients to send back to Stage 1, or NULL on failure
 */
Matrix* stage2_backward(NetworkStage* stage, Matrix* predictions, int* labels, int count, Matrix* stage1_activations, int is_evaluation) {
    if (!stage || !predictions || !labels || count <= 0) return NULL;
    
    /* Create target matrix (one-hot encoded) */
    Matrix* targets = matrix_create(predictions->rows, predictions->cols);
    for (int i = 0; i < count; i++) {
        for (int j = 0; j < predictions->rows; j++) {
            targets->entries[j][i] = (j == labels[i]) ? 1.0 : 0.0;
        }
    }
    
    /* Compute output errors: error = predictions - targets (for softmax + cross-entropy) */
    Matrix* output_errors = matrix_create(predictions->rows, predictions->cols);
    for (int i = 0; i < predictions->rows; i++) {
        for (int j = 0; j < predictions->cols; j++) {
            output_errors->entries[i][j] = predictions->entries[i][j] - targets->entries[i][j];
        }
    }
    
    /* Update Stage 2 weights and biases if stage1 activations available AND not in evaluation mode */
    if (stage1_activations && !is_evaluation) {
        /* Compute weight gradients: dW = output_errors * stage1_activations^T / batch_size */
        Matrix* stage1_activations_T = transpose(stage1_activations);
        Matrix* weight_gradients = dot(output_errors, stage1_activations_T);
        
        /* Update weights: W = W - learning_rate * dW */
        for (int i = 0; i < stage->weights->rows; i++) {
            for (int j = 0; j < stage->weights->cols; j++) {
                stage->weights->entries[i][j] -= stage->learning_rate * weight_gradients->entries[i][j] / count;
            }
        }
        
        /* Update biases: db = mean(output_errors) */
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
    
    /* Compute gradients to send back to Stage 1: gradients = W^T * output_errors */
    Matrix* weights_T = transpose(stage->weights);
    Matrix* gradients_for_stage1 = dot(weights_T, output_errors);
    
    /* Cleanup temporary matrices */
    matrix_free(targets);
    matrix_free(output_errors);
    matrix_free(weights_T);
    
    return gradients_for_stage1;
}

/**
 * @brief Stage 1 backward pass: Compute gradients and update weights
 * 
 * Performs the backward pass for Stage 1, computing gradients with respect to
 * sigmoid activation function. Updates Stage 1 weights and biases if not in evaluation mode.
 * 
 * @param stage Pointer to the Stage 1 network structure
 * @param gradients Gradients received from Stage 2 (512 x batch_size)
 * @param stage1_activations Activations from Stage 1 forward pass (512 x batch_size)
 * @param stage1_inputs Input data for Stage 1 (784 x batch_size)
 * @param is_evaluation Flag indicating evaluation mode (no weight updates)
 * @return NULL (Stage 1 doesn't need to return gradients to previous stage)
 */
Matrix* stage1_backward(NetworkStage* stage, Matrix* gradients, Matrix* stage1_activations, Matrix* stage1_inputs, int is_evaluation) {
    if (!stage || !gradients || !stage1_activations || !stage1_inputs) return NULL;
    
    /* Compute activation derivatives for Stage 1 (sigmoid) */
    Matrix* activation_derivatives = matrix_create(stage1_activations->rows, stage1_activations->cols);
    for (int i = 0; i < stage1_activations->rows; i++) {
        for (int j = 0; j < stage1_activations->cols; j++) {
                double activation = stage1_activations->entries[i][j];
            activation_derivatives->entries[i][j] = activation * (1.0 - activation); /* sigmoid derivative */
        }
    }
    
    /* Apply chain rule: local_gradients = gradients * activation_derivatives (element-wise) */
    Matrix* local_gradients = matrix_create(gradients->rows, gradients->cols);
    for (int i = 0; i < gradients->rows; i++) {
        for (int j = 0; j < gradients->cols; j++) {
            local_gradients->entries[i][j] = gradients->entries[i][j] * activation_derivatives->entries[i][j];
        }
    }
    
    /* Update Stage 1 weights and biases if not in evaluation mode */
    if (!is_evaluation) {
        /* Compute weight gradients: dW = local_gradients * stage1_inputs^T */
        Matrix* stage1_inputs_T = transpose(stage1_inputs);
        Matrix* weight_gradients = dot(local_gradients, stage1_inputs_T);
        
        int batch_size = stage1_inputs->cols;
        
        /* Update weights: W = W - learning_rate * dW / batch_size */
        for (int i = 0; i < stage->weights->rows; i++) {
            for (int j = 0; j < stage->weights->cols; j++) {
                stage->weights->entries[i][j] -= stage->learning_rate * weight_gradients->entries[i][j] / batch_size;
            }
        }
        
        /* Update biases: db = mean(local_gradients) */
        for (int i = 0; i < stage->biases->rows; i++) {
            double bias_gradient = 0.0;
            for (int j = 0; j < local_gradients->cols; j++) {
                bias_gradient += local_gradients->entries[i][j];
            }
            stage->biases->entries[i][0] -= stage->learning_rate * bias_gradient / batch_size;
        }
        
        matrix_free(stage1_inputs_T);
        matrix_free(weight_gradients);
    }
    
    /* Cleanup temporary matrices */
    matrix_free(activation_derivatives);
    matrix_free(local_gradients);
    
    return NULL; /* Stage 1 doesn't need to return gradients to previous stage */
}

void update_stage_weights(NetworkStage* stage, Matrix* gradients) {
    if (!stage || !gradients) return;
    
    // Placeholder cho weight update logic
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
    
    for (int i = 0; i < count; i++) {
        int correct_label = labels[i];
        double predicted_prob = predictions->entries[correct_label][i];
        
        // Cross-entropy loss: -log(predicted_probability)
        if (predicted_prob > 0.0) {
            total_loss += -log(predicted_prob);
        } else {
            total_loss += 10.0; /* Penalty for zero probability */
        }
    }
    
    return total_loss / count; // Average loss
}

int calculate_pipeline_accuracy(Matrix* predictions, int* labels, int count) {
    if (!predictions || !labels || count <= 0) return 0;
    
    int correct = 0;
    
    for (int i = 0; i < count; i++) {
        /* Find class with highest probability */
        int predicted_class = 0;
        double max_prob = predictions->entries[0][i];
        
        for (int j = 1; j < predictions->rows; j++) {
            if (predictions->entries[j][i] > max_prob) {
                max_prob = predictions->entries[j][i];
                predicted_class = j;
            }
        }
        
        if (predicted_class == labels[i]) {
            correct++;
        }
    }
    
    return correct;
}

PipelineStats collect_pipeline_stats(void) {
    PipelineStats stats = {0};
    
    /* Collect stats from global variables with thread safety */
    pthread_mutex_lock(&stats_mutex);
    stats = global_stats;
    pthread_mutex_unlock(&stats_mutex);
    
    return stats;
}

int calculate_optimal_batch_size(PipelineStats* stats) {
    if (!stats) return MINI_BATCH_SIZE;
    
    /* Simple logic to calculate optimal batch size based on statistics */
    double efficiency = stats->pipeline_efficiency;
    double comm_overhead = stats->communication_overhead;
    
    int optimal_size = MINI_BATCH_SIZE;
    
    if (efficiency > 80.0 && comm_overhead < 20.0) {
        optimal_size *= 2; /* Increase batch size if performance is good */
    } else if (efficiency < 50.0 || comm_overhead > 40.0) {
        optimal_size /= 2; /* Decrease batch size if performance is poor */
    }
    
    /* Limit within reasonable range */
    if (optimal_size < 16) optimal_size = 16;
    if (optimal_size > 256) optimal_size = 256;
    
    return optimal_size;
}

long get_current_timestamp(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec; // Microseconds
            }
            
// Phần 2: Triển khai async processing
void* async_forward_processor(void* args) {
    AsyncForwardArgs* fwd_args = (AsyncForwardArgs*)args;
    if (!fwd_args) return NULL;
    
    // Bỏ ghi log async forward
    // printf("[ASYNC_FORWARD] Starting asynchronous forward processor\n");
    // fflush(stdout);
    
    while (*fwd_args->training_active) {
        // Trước tiên, cố gắng xử lý retry queue
        process_retry_queue(fwd_args->buffer);
        
        // Lấy forward message từ buffer
        ForwardMessage* fwd_msg = dequeue_forward(fwd_args->buffer);
        if (!fwd_msg) {
            usleep(1000); // Ngủ 1ms nếu không có messages
            continue;
        }
        
        // Gửi forward activations đến stage tiếp theo
        struct timeval comm_start, comm_end;
        gettimeofday(&comm_start, NULL);
        
        if (send_forward_activations(fwd_args->forward_socket, fwd_msg) < 0) {
            printf("[ASYNC_FORWARD] Lỗi gửi forward message\n");
            free_forward_message(fwd_msg);
            continue;
        }
        
        gettimeofday(&comm_end, NULL);
        double comm_time = get_time_diff(comm_start, comm_end);
        
        // Cập nhật stats giao tiếp
        pthread_mutex_lock(&stats_mutex);
        global_stats.communication_time += comm_time;
        pthread_mutex_unlock(&stats_mutex);
        
        free_forward_message(fwd_msg);
    }
    
    printf("[ASYNC_FORWARD] Dừng async forward processor\n");
    return NULL;
}

void* async_backward_processor(void* args) {
    AsyncBackwardArgs* bwd_args = (AsyncBackwardArgs*)args;
    if (!bwd_args) return NULL;
    
    // Bỏ ghi log async backward
    // printf("[ASYNC_BACKWARD] Starting asynchronous backward processor\n");
    // fflush(stdout);
    
    while (*bwd_args->training_active) {
        // Nhận backward gradients
        BackwardMessage* bwd_msg = receive_backward_gradients_timeout(bwd_args->backward_socket, 100); // Timeout 100ms
        if (!bwd_msg) {
            usleep(1000); // Ngủ 1ms nếu không có messages
            continue;
        }
        
        // GHI LOG METRICS: Kiểm tra messages không theo thứ tự
        // Chỉ ghi log nếu nhận được batch ID cũ hơn ID đã xử lý lần cuối
        if (bwd_msg->batch_id < bwd_args->tracker->last_processed_batch_id) {
            total_out_of_order_messages++;
            extern int current_epoch;
            log_out_of_order_message(bwd_msg->batch_id, bwd_args->tracker->last_processed_batch_id, total_out_of_order_messages, current_epoch);
        }
        
        // Bỏ logging cũ
        // printf("[ASYNC_BACKWARD] Received gradient for batch_id %d\n", bwd_msg->batch_id);
        
        // THAY ĐỔI: Áp dụng gradient trực tiếp thay vì đưa vào queue
        if (apply_gradient_to_pending_batch(bwd_args->stage, bwd_args->tracker, bwd_msg) == 0) {
            // Cập nhật stats loss/accuracy từ gradient
            pthread_mutex_lock(&stats_mutex);
            global_stats.total_loss += bwd_msg->loss;
            global_stats.correct_predictions += bwd_msg->correct_predictions;
            global_stats.total_predictions += bwd_msg->total_predictions;
            pthread_mutex_unlock(&stats_mutex);
            
            // Bỏ logging cũ
            // printf("[ASYNC_BACKWARD] Applied gradient for batch_id %d, loss: %.4f, acc: %d/%d, pending: %d\n", 
            //        bwd_msg->batch_id, bwd_msg->loss, bwd_msg->correct_predictions, bwd_msg->total_predictions,
            //        bwd_args->tracker->pending_count);
        } else {
            // Bỏ logging cũ
            // printf("[ASYNC_BACKWARD] Failed to apply gradient for batch_id %d\n", bwd_msg->batch_id);
        }
        
        free_backward_message(bwd_msg);
        
        // Kiểm tra độ sâu pipeline và điều tiết nếu cần
        if (bwd_args->tracker->pending_count > bwd_args->config->pipeline_depth * 2) {
            printf("[ASYNC_BACKWARD] Pipeline tràn (%d), đang điều tiết...\n", bwd_args->tracker->pending_count);
            usleep(5000); // Điều tiết 5ms
        }
    }
    
    // MỚI: Xử lý gradients còn lại khi shutdown
    printf("[ASYNC_BACKWARD] Xử lý gradients còn lại khi shutdown...\n");
    int cleanup_count = 0;
    while (bwd_args->tracker->pending_count > 0 && cleanup_count < 100) { // Giới hạn cleanup iterations
        BackwardMessage* bwd_msg = receive_backward_gradients_timeout(bwd_args->backward_socket, 1000); // Timeout 1 giây
        if (bwd_msg) {
            apply_gradient_to_pending_batch(bwd_args->stage, bwd_args->tracker, bwd_msg);
            free_backward_message(bwd_msg);
            printf("[ASYNC_BACKWARD] Gradient cleanup đã xử lý, pending: %d\n", bwd_args->tracker->pending_count);
        } else {
            printf("[ASYNC_BACKWARD] Cleanup timeout, còn lại %d batches\n", bwd_args->tracker->pending_count);
            break;
        }
        cleanup_count++;
    }
    
    printf("[ASYNC_BACKWARD] Dừng async backward processor\n");
    return NULL;
}

// Hàm áp dụng gradient bất đồng bộ
int apply_gradient_to_pending_batch(NetworkStage* stage, PipelineBatchTracker* tracker, BackwardMessage* bwd_msg) {
    if (!stage || !tracker || !bwd_msg) return -1;
    
    pthread_mutex_lock(&tracker->tracker_mutex);
    
    // Tìm pending batch tương ứng
    int found_batch = -1;
    for (int i = 0; i < tracker->pending_count; i++) {
        if (tracker->pending_batches[i].batch_id == bwd_msg->batch_id) {
            found_batch = i;
            break;
        }
    }
    
    if (found_batch == -1) {
        // Buffer gradient này để sau nếu batch chưa được tìm thấy
        if (tracker->buffered_count < MAX_PIPELINE_DEPTH) {
            tracker->buffered_gradients[tracker->buffered_count] = malloc(sizeof(BackwardMessage));
            if (tracker->buffered_gradients[tracker->buffered_count]) {
                // Deep copy backward message
                memcpy(tracker->buffered_gradients[tracker->buffered_count], bwd_msg, sizeof(BackwardMessage));
                
                // Deep copy gradients array
                tracker->buffered_gradients[tracker->buffered_count]->gradients = malloc(sizeof(double) * bwd_msg->gradient_count);
                if (tracker->buffered_gradients[tracker->buffered_count]->gradients) {
                    memcpy(tracker->buffered_gradients[tracker->buffered_count]->gradients, 
                           bwd_msg->gradients, sizeof(double) * bwd_msg->gradient_count);
                    tracker->buffered_count++;
                    printf("[ASYNC] Đã buffer gradient cho batch_id %d (chưa được gửi)\n", bwd_msg->batch_id);
                }
            }
        }
        pthread_mutex_unlock(&tracker->tracker_mutex);
        return 0; // Không phải lỗi, chỉ được buffer
    }
    
    BatchContext* batch_ctx = &tracker->pending_batches[found_batch];
    
    // Áp dụng gradient bằng context đã lưu
    int batch_size = (batch_ctx->saved_activations) ? batch_ctx->saved_activations->cols : 32;
    
    // Chuyển gradients trở lại dạng ma trận
    Matrix* gradients_from_stage2 = matrix_create(512, batch_size);
    if (gradients_from_stage2) {
        int idx = 0;
        for (int r = 0; r < 512 && idx < bwd_msg->gradient_count; r++) {
            for (int c = 0; c < batch_size && idx < bwd_msg->gradient_count; c++) {
                gradients_from_stage2->entries[r][c] = bwd_msg->gradients[idx++];
            }
        }
        
        // Thực hiện backpropagation với context đã lưu
        int is_eval_mode = (bwd_msg->batch_id == -1);
        Matrix* stage1_gradients = stage1_backward(stage, gradients_from_stage2, 
                                                  batch_ctx->saved_activations, 
                                                  batch_ctx->saved_inputs, is_eval_mode);
        
        // Dọn dẹp batch đã xử lý
        matrix_free(batch_ctx->saved_activations);
        matrix_free(batch_ctx->saved_inputs);
        batch_ctx->batch_id = -1;
        batch_ctx->saved_activations = NULL;
        batch_ctx->saved_inputs = NULL;
        
        // Nén mảng pending batches
        for (int j = found_batch; j < tracker->pending_count - 1; j++) {
            tracker->pending_batches[j] = tracker->pending_batches[j + 1];
        }
        tracker->pending_count--;
        
        matrix_free(gradients_from_stage2);
        if (stage1_gradients) matrix_free(stage1_gradients);
        
        // GHI LOG METRICS: Cập nhật tracking IDs
        // Luôn tiến tới ID mong đợi tiếp theo sau khi xử lý thành công
        if (bwd_msg->batch_id >= tracker->next_expected_gradient_id) {
            tracker->next_expected_gradient_id = bwd_msg->batch_id + 1;
        }
        // Cập nhật last processed batch ID
        if (bwd_msg->batch_id > tracker->last_processed_batch_id) {
            tracker->last_processed_batch_id = bwd_msg->batch_id;
        }
    }
    
    pthread_mutex_unlock(&tracker->tracker_mutex);
    return 0;
}

int initialize_batch_tracker(PipelineBatchTracker* tracker) {
    if (!tracker) return -1;
    
    tracker->pending_count = 0;
    tracker->next_expected_gradient_id = 0;
    tracker->buffered_count = 0;
    tracker->last_processed_batch_id = -1;
    
    // Initialize all pointers to NULL
    for (int i = 0; i < MAX_PIPELINE_DEPTH; i++) {
        tracker->pending_batches[i].batch_id = -1;
        tracker->pending_batches[i].saved_activations = NULL;
        tracker->pending_batches[i].saved_inputs = NULL;
        tracker->pending_batches[i].timestamp = 0;
        tracker->buffered_gradients[i] = NULL;
    }
    
    // Initialize mutex
    if (pthread_mutex_init(&tracker->tracker_mutex, NULL) != 0) {
        return -1;
    }
    
    return 0;
}

int add_pending_batch(PipelineBatchTracker* tracker, int batch_id, Matrix* activations, Matrix* inputs) {
    if (!tracker || !activations || !inputs) return -1;
    
    pthread_mutex_lock(&tracker->tracker_mutex);
    
    if (tracker->pending_count >= MAX_PIPELINE_DEPTH) {
        pthread_mutex_unlock(&tracker->tracker_mutex);
        return -1; // Buffer full
    }
    
    // Add to pending batches
    int idx = tracker->pending_count;
    tracker->pending_batches[idx].batch_id = batch_id;
    tracker->pending_batches[idx].saved_activations = activations;
    tracker->pending_batches[idx].saved_inputs = inputs;
    tracker->pending_batches[idx].timestamp = get_current_timestamp();
    
    tracker->pending_count++;
    
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
            tracker->pending_batches[i].saved_activations = NULL;
        }
        if (tracker->pending_batches[i].saved_inputs) {
            matrix_free(tracker->pending_batches[i].saved_inputs);
            tracker->pending_batches[i].saved_inputs = NULL;
        }
        tracker->pending_batches[i].batch_id = -1;
    }
    
    // Clean up buffered gradients
    for (int i = 0; i < tracker->buffered_count; i++) {
        if (tracker->buffered_gradients[i]) {
            if (tracker->buffered_gradients[i]->gradients) {
                free(tracker->buffered_gradients[i]->gradients);
            }
            free(tracker->buffered_gradients[i]);
            tracker->buffered_gradients[i] = NULL;
        }
    }
    
    tracker->pending_count = 0;
    tracker->buffered_count = 0;
    
    pthread_mutex_unlock(&tracker->tracker_mutex);
    pthread_mutex_destroy(&tracker->tracker_mutex);
}

int process_pending_gradients(PipelineBatchTracker* tracker, int backward_client) {
    // This function is declared but not used in current implementation
    // Placeholder implementation
    (void)tracker;
    (void)backward_client;
    return 0;
}

 