#pragma once

#include "../matrix/matrix.h"
#include "../util/img.h"
#include <stdbool.h>


// Existing NeuralNetwork structure - ADD these fields
typedef struct {
    int input;
    int hidden; 
    int output;
    double learning_rate;
    Matrix* hidden_weights;
    Matrix* output_weights;
    
    // NEW: Elastic Averaging parameters
    double alpha;           // α - elastic averaging coefficient
    double beta;            // β - server learning rate  
    bool easgd_enabled;     // Enable/disable EASGD
} NeuralNetwork;

// NEW: Elastic Center structure (master-side only)
typedef struct {
    Matrix* center_hidden_weights;   // w̄ hidden
    Matrix* center_output_weights;   // w̄ output
    bool initialized;
    int update_count;
} ElasticCenter;

// NEW: Pipeline Stage Structures
typedef struct {
    int input_size;
    int output_size;
    Matrix* weights;
    double learning_rate;
    bool easgd_enabled;
    double alpha;
    double beta;
} PipelineStage;

typedef struct {
    PipelineStage* stage;
    int stage_id;  // 1 hoặc 2
    int group_id;  // 1 hoặc 2 (data group)
    char* next_stage_ip;
    int next_stage_port;
} PipelineWorker;

// NEW: Separate Elastic Centers for each stage
typedef struct {
    Matrix* center_hidden_weights;   // Stage 1: 784x300
    Matrix* center_output_weights;   // Stage 2: 300x10
    bool hidden_initialized;
    bool output_initialized;
    int hidden_update_count;
    int output_update_count;
} SeparateElasticCenter;

// NEW: Pipeline function declarations
PipelineStage* pipeline_stage_create(int input_size, int output_size, double lr);
void pipeline_stage_free(PipelineStage* stage);
PipelineWorker* pipeline_worker_create(int stage_id, int group_id, 
                                     const char* next_ip, int next_port);
void pipeline_worker_free(PipelineWorker* worker);

NeuralNetwork* network_create(int input, int hidden, int output, double lr);
double network_train(NeuralNetwork* net, Matrix* input, Matrix* output);
double time_in_socket_seconds();
// Thêm tham số epochs vào khai báo hàm
void network_train_batch_imgs(NeuralNetwork* net, Img** imgs, int batch_size, int epochs);
Matrix* network_predict_img(NeuralNetwork* net, Img* img);
double network_predict_imgs(NeuralNetwork* net, Img** imgs, int n);
Matrix* network_predict(NeuralNetwork* net, Matrix* input_data);
void network_save(NeuralNetwork* net, char* file_string);
NeuralNetwork* network_load(char* file_string);
void network_print(NeuralNetwork* net);
void network_free(NeuralNetwork* net);

// NEW: Function declarations
void network_easgd_init(NeuralNetwork* net, double alpha, double beta);
void network_apply_elastic_averaging(NeuralNetwork* net, double* center_weights, int weight_count);
void elastic_center_init(NeuralNetwork* net);
void elastic_center_update(NeuralNetwork* net, double* worker_weights, int weight_count);
void elastic_center_update_sequential(NeuralNetwork* net, double* master_weights, double* slaver_weights, int weight_count);
double* elastic_center_get_weights(int* count_out);
void elastic_center_cleanup(void);

// Stage 1 functions
Matrix* pipeline_stage1_forward(PipelineStage* stage, Matrix* input);
double pipeline_stage1_backward(PipelineStage* stage, Matrix* input, 
                               Matrix* hidden_outputs, Matrix* grad_from_stage2);
double* pipeline_stage1_get_weights(PipelineStage* stage, int* count_out);
void pipeline_stage1_set_weights(PipelineStage* stage, const double* weights, int count);
void pipeline_stage1_apply_elastic_averaging(PipelineStage* stage, double* center_weights, int weight_count);

// Stage 2 functions  
Matrix* pipeline_stage2_forward(PipelineStage* stage, Matrix* hidden_outputs);
double pipeline_stage2_backward(PipelineStage* stage, Matrix* hidden_outputs,
                               Matrix* final_outputs, Matrix* target,
                               Matrix** grad_to_stage1);
double* pipeline_stage2_get_weights(PipelineStage* stage, int* count_out);
void pipeline_stage2_set_weights(PipelineStage* stage, const double* weights, int count);
void pipeline_stage2_apply_elastic_averaging(PipelineStage* stage, double* center_weights, int weight_count);

// Parameter server functions for separate weights
void separate_elastic_center_init_hidden(int hidden_rows, int hidden_cols);
void separate_elastic_center_init_output(int output_rows, int output_cols);
void separate_elastic_center_update_hidden(double* worker_weights, int weight_count, double beta);
void separate_elastic_center_update_output(double* worker_weights, int weight_count, double beta);
double* separate_elastic_center_get_hidden_weights(int* count_out);
double* separate_elastic_center_get_output_weights(int* count_out);
void separate_elastic_center_cleanup(void);

// Network save functions for hybrid parallelism
void network_save_hybrid_final(const char* model_name);
void pipeline_stage_save(PipelineStage* stage, const char* stage_name, int group_id);