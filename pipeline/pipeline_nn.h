/**
 * @file pipeline_nn.h
 * @brief Pipeline Parallelism Neural Network Training System
 * @author Research Team
 * @date 2024
 * @version 1.0
 * 
 * This header file defines the core data structures and function interfaces for
 * a distributed pipeline parallelism system for neural network training. The system
 * implements a two-stage pipeline architecture with asynchronous communication
 * between stages, supporting MNIST digit classification with comprehensive
 * performance metrics collection.
 * 
 * The pipeline architecture consists of:
 * - Stage 1: Input layer (784) -> Hidden layer (512) with sigmoid activation
 * - Stage 2: Hidden layer (512) -> Output layer (10) with softmax activation
 * 
 * Key features:
 * - Asynchronous forward/backward pass processing
 * - Adaptive batch size scaling based on pipeline performance
 * - Comprehensive metrics collection for research analysis
 * - Thread-safe message passing with retry mechanisms
 * - Pipeline bubble detection and mitigation
 */

#ifndef PIPELINE_NN_H
#define PIPELINE_NN_H

#include "../matrix/matrix.h"
#include "../util/img.h"
#include <pthread.h>
#include <sys/time.h>

/* =============================================================================
 * SYSTEM CONSTANTS
 * ============================================================================= */

/** @brief Maximum depth of the processing pipeline for buffering */
#define MAX_PIPELINE_DEPTH 4

/** @brief Default mini-batch size for training */
#define MINI_BATCH_SIZE 32

/* =============================================================================
 * UTILITY MACROS
 * ============================================================================= */

#ifndef min
#define min(a,b) ((a) < (b) ? (a) : (b))
#endif

#ifndef max
#define max(a,b) ((a) > (b) ? (a) : (b))
#endif

/* =============================================================================
 * TIME UTILITY FUNCTIONS
 * ============================================================================= */

/**
 * @brief Calculate time difference between two timeval structures
 * @param start Starting time
 * @param end Ending time
 * @return Time difference in seconds (double precision)
 */
double get_time_diff(struct timeval start, struct timeval end);

/* =============================================================================
 * PIPELINE MESSAGE STRUCTURES
 * ============================================================================= */

/**
 * @brief Forward pass message structure
 * 
 * Contains activation data and labels for forward propagation through the pipeline.
 * Used for communication between Stage 1 and Stage 2.
 */
typedef struct {
    int batch_id;           /**< Unique identifier for this batch */
    int mini_batch_id;      /**< Mini-batch identifier within the batch */
    double* activations;    /**< Activation values from previous stage */
    int activation_count;   /**< Number of activation values */
    int* labels;           /**< Ground truth labels for supervised learning */
    int label_count;       /**< Number of labels */
} ForwardMessage;   

/**
 * @brief Backward pass message structure
 * 
 * Contains gradient information and loss metrics for backward propagation.
 * Used for communication from Stage 2 back to Stage 1.
 */
typedef struct {
    int batch_id;              /**< Unique identifier for this batch */
    int mini_batch_id;         /**< Mini-batch identifier within the batch */
    double* gradients;         /**< Gradient values for backpropagation */
    int gradient_count;        /**< Number of gradient values */
    double loss;               /**< Computed loss value for this batch */
    int correct_predictions;   /**< Number of correct predictions */
    int total_predictions;     /**< Total number of predictions made */
} BackwardMessage;

/* =============================================================================
 * CONTROL MESSAGE STRUCTURES
 * ============================================================================= */

/**
 * @brief Control command enumeration
 * 
 * Defines various control commands for pipeline coordination and optimization.
 */
typedef enum {
    SPEED_UP,    /**< Increase processing speed/batch size */
    SLOW_DOWN,   /**< Decrease processing speed/batch size */
    SYNC,        /**< Synchronize pipeline stages */
    SHUTDOWN     /**< Graceful shutdown command */
} ControlCommand;

/**
 * @brief Control message structure
 * 
 * Used for sending control commands between pipeline stages and coordinator.
 */
typedef struct {
    ControlCommand command;  /**< Command type */
    double parameter;        /**< Optional parameter value */
} ControlMessage;

/* =============================================================================
 * NEURAL NETWORK STAGE STRUCTURE
 * ============================================================================= */

/**
 * @brief Neural network stage structure
 * 
 * Represents a single stage in the pipeline with its weights, biases, and
 * configuration parameters.
 */
typedef struct {
    int stage_id;           /**< Unique identifier for this stage */
    Matrix* weights;        /**< Weight matrix for this layer */
    Matrix* biases;         /**< Bias vector for this layer */
    int input_size;         /**< Number of input features */
    int output_size;        /**< Number of output features */
    double learning_rate;   /**< Learning rate for gradient descent */
} NetworkStage;

/* =============================================================================
 * PIPELINE BUFFER STRUCTURES
 * ============================================================================= */

/**
 * @brief Pipeline buffer for asynchronous processing
 * 
 * Thread-safe circular buffer system for managing forward and backward messages
 * between pipeline stages. Includes retry mechanism for failed transmissions.
 */
typedef struct {
    /* Forward message queue */
    ForwardMessage* forward_queue[MAX_PIPELINE_DEPTH];  /**< Forward message buffer */
    int forward_head;                                   /**< Forward queue head pointer */
    int forward_tail;                                   /**< Forward queue tail pointer */
    
    /* Backward message queue */
    BackwardMessage* backward_queue[MAX_PIPELINE_DEPTH]; /**< Backward message buffer */
    int backward_head;                                   /**< Backward queue head pointer */
    int backward_tail;                                   /**< Backward queue tail pointer */
    
    /* Retry mechanism for failed messages */
    ForwardMessage* retry_queue[MAX_PIPELINE_DEPTH];    /**< Retry message buffer */
    int retry_head;                                     /**< Retry queue head pointer */
    int retry_tail;                                     /**< Retry queue tail pointer */
    int retry_count;                                    /**< Current retry queue size */
    
    /* Thread synchronization primitives */
    pthread_mutex_t forward_mutex;                      /**< Forward queue mutex */
    pthread_mutex_t backward_mutex;                     /**< Backward queue mutex */
    pthread_mutex_t retry_mutex;                        /**< Retry queue mutex */
    pthread_cond_t forward_cond;                        /**< Forward queue condition variable */
    pthread_cond_t backward_cond;                       /**< Backward queue condition variable */
    pthread_cond_t retry_cond;                          /**< Retry queue condition variable */
} PipelineBuffer;

/* =============================================================================
 * PERFORMANCE STATISTICS STRUCTURES
 * ============================================================================= */

/**
 * @brief Comprehensive pipeline performance statistics
 * 
 * Collects detailed performance metrics for research analysis and optimization.
 * Includes timing, throughput, accuracy, and resource utilization metrics.
 */
typedef struct {
    /* Basic performance metrics */
    double stage1_processing_time;      /**< Stage 1 processing time (seconds) */
    double stage2_processing_time;      /**< Stage 2 processing time (seconds) */
    double communication_time;          /**< Inter-stage communication time (seconds) */
    double throughput;                  /**< Overall system throughput (samples/sec) */
    int stage1_queue_size;             /**< Current Stage 1 queue size */
    int stage2_queue_size;             /**< Current Stage 2 queue size */
    
    /* Training metrics */
    double total_loss;                  /**< Accumulated training loss */
    int processed_batches;              /**< Number of processed batches */
    int correct_predictions;            /**< Total correct predictions */
    int total_predictions;              /**< Total predictions made */
    
    /* Advanced pipeline metrics for research */
    double pipeline_efficiency;         /**< Pipeline utilization percentage */
    double communication_overhead;      /**< Communication overhead ratio */
    double memory_usage_mb;            /**< Peak memory usage (MB) */
    double network_bandwidth_mbps;     /**< Average network bandwidth (Mbps) */
    int pipeline_bubbles;              /**< Number of pipeline stalls */
    double gradient_staleness;         /**< Average gradient age (batches) */
    double cpu_utilization;            /**< CPU usage percentage */
    double load_balance_ratio;         /**< Load balance between stages */
    int packets_lost;                  /**< Network packet loss count */
    double jitter_variance;            /**< Network jitter variance */
    double samples_per_second;         /**< Training samples processed per second */
    double convergence_rate;           /**< Loss reduction rate per epoch */
    double energy_efficiency;          /**< Energy efficiency (samples/joule) */
    
    /* Detailed time breakdown */
    double idle_time;                  /**< Pipeline idle time (seconds) */
    double synchronization_time;       /**< Synchronization overhead (seconds) */
    double serialization_time;         /**< Data serialization time (seconds) */
    double queue_wait_time;            /**< Time spent waiting in queues (seconds) */
    
    /* Pipeline depth analysis */
    int max_pipeline_depth_used;      /**< Maximum pipeline depth utilized */
    double avg_pipeline_utilization;   /**< Average pipeline fill ratio */
} PipelineStats;

/**
 * @brief Statistics reporting message
 * 
 * Used for periodic reporting of performance metrics from each stage.
 */
typedef struct {
    int stage_id;                      /**< Reporting stage identifier */
    double processing_time;            /**< Processing time for this report period */
    double communication_time;         /**< Communication time for this period */
    double loss;                       /**< Loss value for this period */
    int batch_count;                   /**< Number of batches processed */
    int correct_predictions;           /**< Correct predictions in this period */
    int total_predictions;             /**< Total predictions in this period */
    long timestamp;                    /**< Timestamp of this report */
} StatsMessage;

/* =============================================================================
 * BATCH TRACKING STRUCTURES
 * ============================================================================= */

/**
 * @brief Batch context for pipeline depth management
 * 
 * Maintains context information for pending batches in the pipeline.
 */
typedef struct {
    int batch_id;                      /**< Batch identifier */
    Matrix* saved_activations;         /**< Saved activation values */
    Matrix* saved_inputs;              /**< Saved input values */
    long timestamp;                    /**< Batch creation timestamp */
} BatchContext;

/**
 * @brief Pipeline batch tracker
 * 
 * Tracks pending batches and manages out-of-order gradient processing.
 */
typedef struct {
    BatchContext pending_batches[MAX_PIPELINE_DEPTH];       /**< Pending batch contexts */
    int pending_count;                                      /**< Number of pending batches */
    int next_expected_gradient_id;                          /**< Next expected gradient ID */
    BackwardMessage* buffered_gradients[MAX_PIPELINE_DEPTH]; /**< Buffered gradient messages */
    int buffered_count;                                     /**< Number of buffered gradients */
    int last_processed_batch_id;                            /**< Last successfully processed batch */
    pthread_mutex_t tracker_mutex;                          /**< Thread safety mutex */
} PipelineBatchTracker;

/* =============================================================================
 * ADAPTIVE PIPELINE CONFIGURATION
 * ============================================================================= */

/**
 * @brief Adaptive pipeline configuration
 * 
 * Dynamic configuration parameters for pipeline optimization.
 */
typedef struct {
    int current_batch_size;            /**< Current mini-batch size */
    int pipeline_depth;                /**< Current pipeline depth */
    double target_latency;             /**< Target latency for operations */
    double communication_ratio;        /**< Communication to computation ratio */
    double bubble_rate;                /**< Current bubble rate percentage */
    long last_adjustment_time;         /**< Last configuration adjustment time */
} PipelineConfig;

/* =============================================================================
 * THREAD ARGUMENT STRUCTURES
 * ============================================================================= */

/**
 * @brief Arguments for asynchronous forward processing thread
 */
typedef struct {
    NetworkStage* stage;               /**< Neural network stage */
    PipelineBuffer* buffer;            /**< Pipeline buffer */
    PipelineBatchTracker* tracker;     /**< Batch tracker */
    int forward_socket;                /**< Forward communication socket */
    int* training_active;              /**< Training status flag */
    PipelineConfig* config;            /**< Pipeline configuration */
} AsyncForwardArgs;

/**
 * @brief Arguments for asynchronous backward processing thread
 */
typedef struct {
    NetworkStage* stage;               /**< Neural network stage */
    PipelineBuffer* buffer;            /**< Pipeline buffer */
    PipelineBatchTracker* tracker;     /**< Batch tracker */
    int backward_socket;               /**< Backward communication socket */
    int* training_active;              /**< Training status flag */
    PipelineConfig* config;            /**< Pipeline configuration */
} AsyncBackwardArgs;

/* =============================================================================
 * EXPERIMENT RESULTS STRUCTURE
 * ============================================================================= */

/**
 * @brief Comprehensive experiment results for research publication
 * 
 * Aggregates all experimental data for academic analysis and reporting.
 */
typedef struct {
    char experiment_name[64];          /**< Experiment identifier */
    int total_epochs;                  /**< Total training epochs */
    int total_samples;                 /**< Total samples processed */
    double total_training_time;        /**< Total training time (seconds) */
    double baseline_time;              /**< Sequential baseline time (seconds) */
    double speedup_factor;             /**< Parallel speedup factor */
    double efficiency;                 /**< Parallel efficiency */
    
    /* Network condition impact analysis */
    char network_type[16];             /**< Network type (LAN/WAN/SLOW) */
    double network_latency_ms;         /**< Network latency (milliseconds) */
    double network_bandwidth_mbps;     /**< Network bandwidth (Mbps) */
    double packet_loss_rate;           /**< Packet loss rate percentage */
    
    /* Scalability analysis */
    int num_pipeline_stages;           /**< Number of pipeline stages */
    double scalability_factor;         /**< Performance improvement per stage */
    
    PipelineStats final_stats;         /**< Final aggregated statistics */
} ExperimentResults;

/* =============================================================================
 * FUNCTION DECLARATIONS
 * ============================================================================= */

/* Buffer management functions */
int initialize_buffer(PipelineBuffer* buffer);
int enqueue_forward(PipelineBuffer* buffer, ForwardMessage* msg);
ForwardMessage* dequeue_forward(PipelineBuffer* buffer);
int enqueue_backward(PipelineBuffer* buffer, BackwardMessage* msg);
BackwardMessage* dequeue_backward(PipelineBuffer* buffer);

/* Retry queue management */
int enqueue_retry(PipelineBuffer* buffer, ForwardMessage* msg);
ForwardMessage* dequeue_retry(PipelineBuffer* buffer);
int get_retry_count(PipelineBuffer* buffer);
void process_retry_queue(PipelineBuffer* buffer);
void cleanup_buffer(PipelineBuffer* buffer);

/* Message handling functions */
ForwardMessage* create_forward_message(int batch_id, int mini_batch_id, 
                                     Matrix* activations, int* labels, int label_count);
BackwardMessage* create_backward_message(int batch_id, int mini_batch_id, 
                                       Matrix* gradients, double loss, int correct, int total);
void free_forward_message(ForwardMessage* msg);
void free_backward_message(BackwardMessage* msg);

/* Network communication functions */
int send_forward_activations(int sockfd, ForwardMessage* msg);
ForwardMessage* receive_forward_activations(int sockfd);
int send_backward_gradients(int sockfd, BackwardMessage* msg);
BackwardMessage* receive_backward_gradients(int sockfd);
BackwardMessage* receive_backward_gradients_timeout(int sockfd, int timeout_ms);
BackwardMessage* receive_backward_gradients_nonblocking(int sockfd);
int send_control_message(int sockfd, ControlMessage* msg);
ControlMessage* receive_control_message(int sockfd);

/* Statistics communication functions */
int send_stats_message(int sockfd, StatsMessage* msg);
StatsMessage* receive_stats_message(int sockfd);
int connect_stats_client(const char* coordinator_ip, int stats_port);
void stats_server_main(int stats_port, PipelineStats* aggregated_stats);

/* Neural network stage functions */
NetworkStage* create_stage(int stage_id, int input_size, int output_size, double lr);
Matrix* stage1_forward(NetworkStage* stage, Img** imgs, int count);
Matrix* stage2_forward(NetworkStage* stage, Matrix* activations);
Matrix* stage1_backward(NetworkStage* stage, Matrix* gradients, Matrix* stage1_activations, 
                       Matrix* stage1_inputs, int is_evaluation);
Matrix* stage2_backward(NetworkStage* stage, Matrix* predictions, int* labels, int count, 
                       Matrix* stage1_activations, int is_evaluation);
void update_stage_weights(NetworkStage* stage, Matrix* gradients);
void free_stage(NetworkStage* stage);

/* Asynchronous processing functions */
void* async_forward_processor(void* args);
void* async_backward_processor(void* args);

/* Pipeline main functions */
void pipeline_stage1_main(int forward_port, int backward_port);
void pipeline_stage2_main(int forward_port, int backward_port, const char* stage1_ip);
void coordinator_main(int stage1_port, int stage2_port);

/* Evaluation functions */
void pipeline_evaluate(int forward_port, int backward_port, const char* stage1_ip, 
                      const char* test_dataset_path);

/* Performance analysis functions */
double calculate_pipeline_loss(Matrix* predictions, int* labels, int count);
int calculate_pipeline_accuracy(Matrix* predictions, int* labels, int count);
PipelineStats collect_pipeline_stats(void);
int calculate_optimal_batch_size(PipelineStats* stats);

/* Batch tracking functions */
int initialize_batch_tracker(PipelineBatchTracker* tracker);
int add_pending_batch(PipelineBatchTracker* tracker, int batch_id, 
                     Matrix* activations, Matrix* inputs);
int process_pending_gradients(PipelineBatchTracker* tracker, int backward_client);
void cleanup_batch_tracker(PipelineBatchTracker* tracker);
long get_current_timestamp(void);

/* Adaptive pipeline configuration functions */
void update_pipeline_config(PipelineConfig* config, PipelineStats* stats);
int calculate_adaptive_batch_size(PipelineStats* stats, PipelineConfig* config);
void adjust_pipeline_depth(PipelineConfig* config, PipelineStats* stats);
void analyze_bubble_rate_trend(PipelineConfig* config, PipelineStats* stats);

/* Gradient processing functions */
int apply_single_gradient(NetworkStage* stage, PipelineBatchTracker* tracker, 
                         BackwardMessage* bwd_msg);
int apply_gradient_to_pending_batch(NetworkStage* stage, PipelineBatchTracker* tracker, 
                                   BackwardMessage* bwd_msg);
int process_available_gradients_nonblocking(PipelineBatchTracker* tracker, 
                                           NetworkStage* stage, int backward_client);
int process_gradients_with_pipeline_control(PipelineBatchTracker* tracker, 
                                           NetworkStage* stage, int backward_client, 
                                           int max_pipeline_depth);

/* Queue size monitoring functions */
int get_forward_queue_size(PipelineBuffer* buffer);
int get_backward_queue_size(PipelineBuffer* buffer);

/* Logging and monitoring functions */
void log_queue_sizes(PipelineBuffer* buffer, int batch_count, int epoch);
void log_bubble_rate(int total_bubbles, int batch_count, int epoch);
void log_out_of_order_message(int batch_id, int expected_id, int total_out_of_order, int epoch);

#endif /* PIPELINE_NN_H */ 