#include "pipeline_utils.h"
#include "../matrix/ops.h"
#include "../neural/nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Basic activation functions
double sigmoid_activation(double x) {
    return 1.0 / (1.0 + exp(-x));
}

Matrix* apply_sigmoid(Matrix* m) {
    Matrix* result = matrix_create(m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            result->entries[i][j] = sigmoid_activation(m->entries[i][j]);
        }
    }
    return result;
}

Matrix* sigmoid_derivative(Matrix* m) {
    Matrix* result = matrix_create(m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            double s = m->entries[i][j];
            result->entries[i][j] = s * (1.0 - s);
        }
    }
    return result;
}

Matrix* matrix_multiply_elementwise(Matrix* a, Matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        printf("Matrix dimension mismatch in elementwise multiply\n");
        return NULL;
    }
    
    Matrix* result = matrix_create(a->rows, a->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            result->entries[i][j] = a->entries[i][j] * b->entries[i][j];
        }
    }
    return result;
}

Matrix* matrix_transpose_multiply(Matrix* a, Matrix* b) {
    // Compute a^T * b
    Matrix* a_transpose = transpose(a);
    Matrix* result = dot(a_transpose, b);
    matrix_free(a_transpose);
    return result;
}

// ReLU activation (alternative to sigmoid)
Matrix* relu_activation(Matrix* input) {
    Matrix* result = matrix_create(input->rows, input->cols);
    for (int i = 0; i < input->rows; i++) {
        for (int j = 0; j < input->cols; j++) {
            result->entries[i][j] = fmax(0.0, input->entries[i][j]);
        }
    }
    return result;
}

Matrix* relu_derivative(Matrix* input) {
    Matrix* result = matrix_create(input->rows, input->cols);
    for (int i = 0; i < input->rows; i++) {
        for (int j = 0; j < input->cols; j++) {
            result->entries[i][j] = (input->entries[i][j] > 0.0) ? 1.0 : 0.0;
        }
    }
    return result;
}

Matrix* softmax_activation(Matrix* input) {
    Matrix* result = matrix_create(input->rows, input->cols);
    
    // Find max value for numerical stability
    double max_val = input->entries[0][0];
    for (int i = 0; i < input->rows; i++) {
        for (int j = 0; j < input->cols; j++) {
            if (input->entries[i][j] > max_val) {
                max_val = input->entries[i][j];
            }
        }
    }
    
    // Compute exponentials and sum
    double sum = 0.0;
    for (int i = 0; i < input->rows; i++) {
        for (int j = 0; j < input->cols; j++) {
            result->entries[i][j] = exp(input->entries[i][j] - max_val);
            sum += result->entries[i][j];
        }
    }
    
    // Normalize
    for (int i = 0; i < input->rows; i++) {
        for (int j = 0; j < input->cols; j++) {
            result->entries[i][j] /= sum;
        }
    }
    
    return result;
}

// Pipeline Stage implementations
PipelineStage* pipeline_stage_create(int input_size, int output_size, double lr) {
    PipelineStage* stage = malloc(sizeof(PipelineStage));
    stage->input_size = input_size;
    stage->output_size = output_size;
    stage->learning_rate = lr;
    stage->easgd_enabled = false;
    stage->alpha = 0.0;
    stage->beta = 0.0;
    
    // Initialize weights with Xavier initialization
    stage->weights = matrix_create(output_size, input_size);
    matrix_randomize(stage->weights, input_size);
    
    return stage;
}

void pipeline_stage_free(PipelineStage* stage) {
    if (stage) {
        matrix_free(stage->weights);
        free(stage);
    }
}

PipelineWorker* pipeline_worker_create(int stage_id, int group_id, 
                                     const char* next_ip, int next_port) {
    PipelineWorker* worker = malloc(sizeof(PipelineWorker));
    worker->stage_id = stage_id;
    worker->group_id = group_id;
    worker->next_stage_ip = malloc(strlen(next_ip) + 1);
    strcpy(worker->next_stage_ip, next_ip);
    worker->next_stage_port = next_port;
    worker->stage = NULL; // Will be set later
    
    return worker;
}

void pipeline_worker_free(PipelineWorker* worker) {
    if (worker) {
        if (worker->next_stage_ip) {
            free(worker->next_stage_ip);
        }
        if (worker->stage) {
            pipeline_stage_free(worker->stage);
        }
        free(worker);
    }
}

// Stage 1 functions (784 → 300)
Matrix* pipeline_stage1_forward(PipelineStage* stage, Matrix* input) {
    // Forward pass: activation = sigmoid(W * input)
    Matrix* z = dot(stage->weights, input);  // 300x784 * 784x1 = 300x1
    Matrix* activation = apply_sigmoid(z);
    matrix_free(z);
    return activation;
}

double pipeline_stage1_backward(PipelineStage* stage, Matrix* input, Matrix* grad_from_stage2) {
    // Forward pass to get intermediate values
    Matrix* z = dot(stage->weights, input);
    Matrix* activation = apply_sigmoid(z);
    
    // Backward pass
    Matrix* sigmoid_grad = sigmoid_derivative(activation);
    Matrix* delta = matrix_multiply_elementwise(grad_from_stage2, sigmoid_grad);
    
    // Compute gradients for weights
    Matrix* input_transpose = transpose(input);
    Matrix* weight_grad = dot(delta, input_transpose);
    
    // Update weights
    Matrix* scaled_grad = scale(stage->learning_rate, weight_grad);
    Matrix* new_weights = subtract(stage->weights, scaled_grad);
    
    matrix_free(stage->weights);
    stage->weights = new_weights;
    
    // Compute loss (MSE with gradient)
    double loss = 0.0;
    for (int i = 0; i < grad_from_stage2->rows; i++) {
        loss += grad_from_stage2->entries[i][0] * grad_from_stage2->entries[i][0];
    }
    loss /= grad_from_stage2->rows;
    
    // Cleanup
    matrix_free(z);
    matrix_free(activation);
    matrix_free(sigmoid_grad);
    matrix_free(delta);
    matrix_free(input_transpose);
    matrix_free(weight_grad);
    matrix_free(scaled_grad);
    
    return loss;
}

double* pipeline_stage1_get_weights(PipelineStage* stage, int* count_out) {
    int count = stage->weights->rows * stage->weights->cols;
    double* weights = malloc(sizeof(double) * count);
    
    int idx = 0;
    for (int i = 0; i < stage->weights->rows; i++) {
        for (int j = 0; j < stage->weights->cols; j++) {
            weights[idx++] = stage->weights->entries[i][j];
        }
    }
    
    *count_out = count;
    return weights;
}

void pipeline_stage1_set_weights(PipelineStage* stage, const double* weights, int count) {
    int idx = 0;
    for (int i = 0; i < stage->weights->rows; i++) {
        for (int j = 0; j < stage->weights->cols; j++) {
            stage->weights->entries[i][j] = weights[idx++];
        }
    }
}

void pipeline_stage1_apply_elastic_averaging(PipelineStage* stage, double* center_weights, int weight_count) {
    if (!stage->easgd_enabled) {
        pipeline_stage1_set_weights(stage, center_weights, weight_count);
        return;
    }
    
    // Apply elastic averaging: w ← w + α(w̄ - w)
    Matrix* center_matrix = matrix_create(stage->weights->rows, stage->weights->cols);
    
    int idx = 0;
    for (int i = 0; i < center_matrix->rows; i++) {
        for (int j = 0; j < center_matrix->cols; j++) {
            center_matrix->entries[i][j] = center_weights[idx++];
        }
    }
    
    Matrix* diff = subtract(center_matrix, stage->weights);
    Matrix* update = scale(stage->alpha, diff);
    Matrix* new_weights = add(stage->weights, update);
    
    matrix_free(stage->weights);
    stage->weights = new_weights;
    
    matrix_free(center_matrix);
    matrix_free(diff);
    matrix_free(update);
}

// Stage 2 functions (300 → 10)
Matrix* pipeline_stage2_forward(PipelineStage* stage, Matrix* hidden_activation) {
    Matrix* z = dot(stage->weights, hidden_activation);  // 10x300 * 300x1 = 10x1
    Matrix* output = apply_sigmoid(z);
    matrix_free(z);
    return output;
}

double pipeline_stage2_backward(PipelineStage* stage, Matrix* hidden_activation, Matrix* target, Matrix** grad_to_stage1) {
    // Forward pass
    Matrix* z = dot(stage->weights, hidden_activation);
    Matrix* output = apply_sigmoid(z);
    
    // Compute output error and loss
    Matrix* output_error = subtract(target, output);
    double loss = 0.0;
    for (int i = 0; i < output_error->rows; i++) {
        double err = output_error->entries[i][0];
        loss += err * err;
    }
    loss /= output_error->rows;
    
    // Backward pass
    Matrix* sigmoid_grad = sigmoid_derivative(output);
    Matrix* delta_output = matrix_multiply_elementwise(output_error, sigmoid_grad);
    
    // Update output weights
    Matrix* hidden_transpose = transpose(hidden_activation);
    Matrix* weight_grad = dot(delta_output, hidden_transpose);
    Matrix* scaled_grad = scale(stage->learning_rate, weight_grad);
    Matrix* new_weights = add(stage->weights, scaled_grad);
    
    matrix_free(stage->weights);
    stage->weights = new_weights;
    
    // Compute gradient to send back to stage 1
    Matrix* weights_transpose = transpose(stage->weights);
    *grad_to_stage1 = dot(weights_transpose, delta_output);
    
    // Cleanup
    matrix_free(z);
    matrix_free(output);
    matrix_free(output_error);
    matrix_free(sigmoid_grad);
    matrix_free(delta_output);
    matrix_free(hidden_transpose);
    matrix_free(weight_grad);
    matrix_free(scaled_grad);
    matrix_free(weights_transpose);
    
    return loss;
}

double* pipeline_stage2_get_weights(PipelineStage* stage, int* count_out) {
    int count = stage->weights->rows * stage->weights->cols;
    double* weights = malloc(sizeof(double) * count);
    
    int idx = 0;
    for (int i = 0; i < stage->weights->rows; i++) {
        for (int j = 0; j < stage->weights->cols; j++) {
            weights[idx++] = stage->weights->entries[i][j];
        }
    }
    
    *count_out = count;
    return weights;
}

void pipeline_stage2_set_weights(PipelineStage* stage, const double* weights, int count) {
    int idx = 0;
    for (int i = 0; i < stage->weights->rows; i++) {
        for (int j = 0; j < stage->weights->cols; j++) {
            stage->weights->entries[i][j] = weights[idx++];
        }
    }
}

void pipeline_stage2_apply_elastic_averaging(PipelineStage* stage, double* center_weights, int weight_count) {
    if (!stage->easgd_enabled) {
        pipeline_stage2_set_weights(stage, center_weights, weight_count);
        return;
    }
    
    // Apply elastic averaging: w ← w + α(w̄ - w)
    Matrix* center_matrix = matrix_create(stage->weights->rows, stage->weights->cols);
    
    int idx = 0;
    for (int i = 0; i < center_matrix->rows; i++) {
        for (int j = 0; j < center_matrix->cols; j++) {
            center_matrix->entries[i][j] = center_weights[idx++];
        }
    }
    
    Matrix* diff = subtract(center_matrix, stage->weights);
    Matrix* update = scale(stage->alpha, diff);
    Matrix* new_weights = add(stage->weights, update);
    
    matrix_free(stage->weights);
    stage->weights = new_weights;
    
    matrix_free(center_matrix);
    matrix_free(diff);
    matrix_free(update);
}