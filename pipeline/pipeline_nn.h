#ifndef PIPELINE_NN_H
#define PIPELINE_NN_H

#include "../matrix/matrix.h"
#include "../util/img.h"
#include <pthread.h>

#define MAX_PIPELINE_DEPTH 8
#define MINI_BATCH_SIZE 32

// Pipeline message structures
typedef struct {
    int batch_id;
    int mini_batch_id;
    double* activations;
    int activation_count;
    int* labels;
    int label_count;
} ForwardMessage;

typedef struct {
    int batch_id;
    int mini_batch_id;
    double* gradients;
    int gradient_count;
    double loss;
    int correct_predictions;
    int total_predictions;
} BackwardMessage;

// Control messages
typedef enum {
    SPEED_UP,
    SLOW_DOWN,
    SYNC,
    SHUTDOWN
} ControlCommand;

typedef struct {
    ControlCommand command;
    double parameter;
} ControlMessage;

// Network stage structure
typedef struct {
    int stage_id;
    Matrix* weights;
    Matrix* biases;
    int input_size;
    int output_size;
    double learning_rate;
} NetworkStage;

// Pipeline buffer for async processing
typedef struct {
    ForwardMessage* forward_queue[MAX_PIPELINE_DEPTH];
    BackwardMessage* backward_queue[MAX_PIPELINE_DEPTH];
    int forward_head, forward_tail;
    int backward_head, backward_tail;
    pthread_mutex_t forward_mutex;
    pthread_mutex_t backward_mutex;
    pthread_cond_t forward_cond;
    pthread_cond_t backward_cond;
} PipelineBuffer;

// Pipeline statistics
typedef struct {
    double stage1_processing_time;
    double stage2_processing_time;
    double communication_time;
    double throughput;
    int stage1_queue_size;
    int stage2_queue_size;
    double total_loss;
    int processed_batches;
    int correct_predictions;
    int total_predictions;
} PipelineStats;

// Stats reporting message
typedef struct {
    int stage_id; // 1 for stage1, 2 for stage2
    double processing_time;
    double communication_time;
    double loss;
    int batch_count;
    int correct_predictions;
    int total_predictions;
    long timestamp;
} StatsMessage;

// Function declarations
// Buffer management
int initialize_buffer(PipelineBuffer* buffer);
int enqueue_forward(PipelineBuffer* buffer, ForwardMessage* msg);
ForwardMessage* dequeue_forward(PipelineBuffer* buffer);
int enqueue_backward(PipelineBuffer* buffer, BackwardMessage* msg);
BackwardMessage* dequeue_backward(PipelineBuffer* buffer);
void cleanup_buffer(PipelineBuffer* buffer);

// Message handling
ForwardMessage* create_forward_message(int batch_id, int mini_batch_id, Matrix* activations, int* labels, int label_count);
BackwardMessage* create_backward_message(int batch_id, int mini_batch_id, Matrix* gradients, double loss, int correct, int total);
void free_forward_message(ForwardMessage* msg);
void free_backward_message(BackwardMessage* msg);

// Network communication
int send_forward_activations(int sockfd, ForwardMessage* msg);
ForwardMessage* receive_forward_activations(int sockfd);
int send_backward_gradients(int sockfd, BackwardMessage* msg);
BackwardMessage* receive_backward_gradients(int sockfd);
int send_control_message(int sockfd, ControlMessage* msg);
ControlMessage* receive_control_message(int sockfd);

// Stats communication
int send_stats_message(int sockfd, StatsMessage* msg);
StatsMessage* receive_stats_message(int sockfd);
int connect_stats_client(const char* coordinator_ip, int stats_port);
void stats_server_main(int stats_port, PipelineStats* aggregated_stats);

// Stage processing
NetworkStage* create_stage(int stage_id, int input_size, int output_size, double lr);
Matrix* stage1_forward(NetworkStage* stage, Img** imgs, int count);
Matrix* stage2_forward(NetworkStage* stage, Matrix* activations);
Matrix* stage1_backward(NetworkStage* stage, Matrix* gradients, Matrix* stage1_activations, Matrix* stage1_inputs, int is_evaluation);
Matrix* stage2_backward(NetworkStage* stage, Matrix* predictions, int* labels, int count, Matrix* stage1_activations, int is_evaluation);
void update_stage_weights(NetworkStage* stage, Matrix* gradients);
void free_stage(NetworkStage* stage);

// Pipeline main functions
void pipeline_stage1_main(int forward_port, int backward_port);
void pipeline_stage2_main(int forward_port, int backward_port, const char* stage1_ip);
void coordinator_main(int stage1_port, int stage2_port);

// Evaluation functions
void pipeline_evaluate(int forward_port, int backward_port, const char* stage1_ip, const char* test_dataset_path);

// Global stats access
extern PipelineStats global_stats;
extern pthread_mutex_t stats_mutex;

// Utility functions
double calculate_pipeline_loss(Matrix* predictions, int* labels, int count);
int calculate_pipeline_accuracy(Matrix* predictions, int* labels, int count);
PipelineStats collect_pipeline_stats(void);
int calculate_optimal_batch_size(PipelineStats* stats);

#endif 