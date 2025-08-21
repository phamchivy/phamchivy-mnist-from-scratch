#include "pipeline_utils.h"
#include "../matrix/ops.h"
#include "../neural/nn.h"
#include "../neural/activations.h"
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
    Matrix* hidden_inputs = dot(stage->weights, input);    // W1 × input
    Matrix* hidden_outputs = apply(sigmoid, hidden_inputs); // σ(W1 × input)
    
    matrix_free(hidden_inputs);
    return hidden_outputs;  // Return activation để gửi cho Stage 2
}

// Backward: chỉ làm backward pass với pre-computed activations
double pipeline_stage1_backward(PipelineStage* stage, Matrix* input, 
                               Matrix* hidden_outputs, Matrix* grad_from_stage2) {
    // Không làm forward nữa - dùng hidden_outputs đã có!
    
    // Backward pass theo network_train() chuẩn
    Matrix* sigmoid_primed_mat = sigmoidPrime(hidden_outputs);
    Matrix* multiplied_mat = multiply(grad_from_stage2, sigmoid_primed_mat);
    Matrix* transposed_mat = transpose(input);
    Matrix* dot_mat = dot(multiplied_mat, transposed_mat);
    Matrix* scaled_mat = scale(stage->learning_rate, dot_mat);
    Matrix* added_mat = add(stage->weights, scaled_mat);  // FIXED: ADD not subtract
    
    matrix_free(stage->weights);
    stage->weights = added_mat;
    
    // Cleanup
    matrix_free(sigmoid_primed_mat);
    matrix_free(multiplied_mat);
    matrix_free(transposed_mat);
    matrix_free(dot_mat);
    matrix_free(scaled_mat);
    
    return 0.0;  // Stage1 không có meaningful loss
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
Matrix* pipeline_stage2_forward(PipelineStage* stage, Matrix* hidden_outputs) {
    Matrix* final_inputs = dot(stage->weights, hidden_outputs);   // W2 × hidden
    Matrix* final_outputs = apply(sigmoid, final_inputs);         // σ(W2 × hidden)
    
    matrix_free(final_inputs);
    return final_outputs;  // Return prediction
}

double pipeline_stage2_backward(PipelineStage* stage, Matrix* hidden_outputs,
                               Matrix* final_outputs, Matrix* target,
                               Matrix** grad_to_stage1) {
    // Không làm forward nữa - dùng final_outputs đã có!
    
    // Compute loss theo network_train() chuẩn
    Matrix* output_errors = subtract(target, final_outputs);
    double loss = 0.0;
    for (int i = 0; i < target->rows; i++) {
        double diff = target->entries[i][0] - final_outputs->entries[i][0];
        loss += diff * diff;
    }
    
    // Backward pass theo network_train() chuẩn
    Matrix* sigmoid_primed_mat = sigmoidPrime(final_outputs);
    Matrix* multiplied_mat = multiply(output_errors, sigmoid_primed_mat);
    Matrix* transposed_mat = transpose(hidden_outputs);
    Matrix* dot_mat = dot(multiplied_mat, transposed_mat);
    Matrix* scaled_mat = scale(stage->learning_rate, dot_mat);
    Matrix* added_mat = add(stage->weights, scaled_mat);
    
    matrix_free(stage->weights);
    stage->weights = added_mat;
    
    // Compute gradient to send back (theo network_train() chuẩn)
    Matrix* weights_transpose = transpose(stage->weights);  // ⚠️ Dùng OLD weights!
    *grad_to_stage1 = dot(weights_transpose, output_errors);
    
    // Cleanup
    matrix_free(output_errors);
    matrix_free(sigmoid_primed_mat);
    matrix_free(multiplied_mat);
    matrix_free(transposed_mat);
    matrix_free(dot_mat);
    matrix_free(scaled_mat);
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