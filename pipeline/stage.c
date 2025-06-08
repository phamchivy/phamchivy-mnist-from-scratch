#include "pipeline_nn.h"
#include "../neural/activations.h"
#include "../matrix/ops.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

// Global variables for loss calculation
static double current_total_loss = 0.0;
static int current_processed_batches = 0;
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