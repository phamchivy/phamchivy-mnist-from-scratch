/**
 * @file hybrid_pipeline_easgd.h
 * @brief Hybrid Pipeline Parallelism + EASGD Neural Network Training System
 * @author Research Team
 * @date 2024
 * @version 2.0
 * 
 * Extended header for hybrid system combining Pipeline Parallelism with
 * Elastic Averaging SGD (EASGD) for distributed neural network training.
 * 
 * Hybrid Architecture:
 * - Stage1-1: Data[1-30K] → Pipeline → Stage2-1 (with α₁ parameter)
 * - Stage1-2: Data[30K-60K] → Pipeline → Stage2-1 (with α₁ parameter)  
 * - Stage2-1: Multiplexed processing (with α₂ parameter)
 * - Coordinator: EASGD Parameter Server (with β parameter)
 */

#ifndef HYBRID_PIPELINE_EASGD_H
#define HYBRID_PIPELINE_EASGD_H

#include "pipeline_nn.h"  // Include original pipeline definitions
#include <sys/select.h>
#include <errno.h>
#include <stdlib.h>      // malloc, free, getenv
#include <stdio.h>       // printf, fprintf, fopen, fclose, FILE, stdout, fflush
#include <string.h>      // memset, strcpy, strcmp
#include <netinet/in.h>  // struct sockaddr_in, socklen_t
#include <sys/socket.h>  // accept

/* =============================================================================
 * EASGD SYSTEM CONSTANTS
 * ============================================================================= */

/** @brief Maximum number of workers in EASGD system */
#define MAX_EASGD_WORKERS 4

/** @brief Maximum number of parameters in flattened weight vector */
#define MAX_WEIGHT_PARAMETERS 500000

/** @brief EASGD communication port */
#define EASGD_COMM_PORT 12348

/** @brief Default EASGD hyperparameters */
#define DEFAULT_ALPHA_STAGE1 0.2    /**< Default α₁ for Stage1 workers */
#define DEFAULT_ALPHA_STAGE2 0.3    /**< Default α₂ for Stage2 worker */
#define DEFAULT_BETA 0.5            /**< Default β for parameter server */

/** @brief Communication timeout in seconds */
#define EASGD_TIMEOUT_SEC 30

/* =============================================================================
 * EASGD MESSAGE STRUCTURES
 * ============================================================================= */

/**
 * @brief EASGD message types for communication protocol
 */
typedef enum {
    EASGD_WEIGHT_UPDATE,        /**< Send local weights to parameter server */
    EASGD_MASTER_WEIGHTS,       /**< Send master weights to workers */
    EASGD_SYNC_REQUEST,         /**< Request synchronization */
    EASGD_EPOCH_COMPLETE,       /**< Signal epoch completion */
    EASGD_TRAINING_COMPLETE,    /**< Signal training completion */
    EASGD_EVALUATION_MODE       /**< Switch to evaluation mode */
} EASGDMessageType;

/**
 * @brief EASGD communication message structure
 */
typedef struct {
    EASGDMessageType message_type;  /**< Message type identifier */
    int worker_id;                  /**< Worker identification (1-4) */
    int stage_type;                 /**< Stage type: 1 for Stage1, 2 for Stage2 */
    int epoch;                      /**< Current epoch number */
    int communication_round;        /**< EASGD communication round */
    
    /* Weight data */
    int stage1_weight_count;        /**< Number of Stage1 parameters */
    int stage2_weight_count;        /**< Number of Stage2 parameters */
    double* stage1_weights;         /**< Flattened Stage1 weights */
    double* stage2_weights;         /**< Flattened Stage2 weights */
    
    /* Optional metadata */
    double local_loss;              /**< Worker's current loss */
    double local_accuracy;          /**< Worker's current accuracy */
    int processed_samples;          /**< Samples processed by worker */
    long timestamp;                 /**< Message timestamp */
} EASGDMessage;

/* =============================================================================
 * MESSAGE QUEUE STRUCTURE
 * ============================================================================= */

/**
 * @brief Thread-safe message queue for multiplexing
 */
typedef struct {
    ForwardMessage* messages[MAX_PIPELINE_DEPTH];
    int head, tail, count;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} MessageQueue;

/* =============================================================================
 * HYBRID WORKER STRUCTURES
 * ============================================================================= */

/**
 * @brief Stage1 Worker structure for hybrid system
 */
typedef struct {
    /* Worker identification */
    int worker_id;                  /**< Worker ID (1 or 2) */
    int data_start_idx;             /**< Start index in dataset */
    int data_end_idx;               /**< End index in dataset */
    
    /* Pipeline components (inherited from original) */
    NetworkStage* stage1;           /**< Local Stage1 network: 784 → 512 */
    PipelineBuffer pipeline_buffer; /**< Async pipeline buffer */
    PipelineBatchTracker batch_tracker; /**< Batch tracking system */
    PipelineConfig config;          /**< Adaptive pipeline configuration */
    
    /* EASGD parameters */
    double alpha1;                  /**< Elastic averaging rate for Stage1 */
    double* stage1_weights_flat;    /**< Flattened Stage1 weights for EASGD */
    int stage1_weight_count;        /**< Number of Stage1 parameters */
    
    /* Pipeline communication sockets */
    int stage2_forward_socket;      /**< Send activations to Stage2-1 */
    int stage2_backward_socket;     /**< Receive gradients from Stage2-1 */
    int stage2_backward_server;     /**< Backward server for Stage2 to connect */
    
    /* EASGD communication */
    int last_easgd_socket;          /**< Short-lived EASGD socket per epoch */
    
    /* Async processing threads */
    pthread_t forward_thread;       /**< Async forward processing thread */
    pthread_t backward_thread;      /**< Async backward processing thread */
    int* training_active;           /**< Shared training status flag */
    
    /* Performance metrics */
    PipelineStats local_stats;      /**< Local pipeline statistics */
    double total_easgd_comm_time;   /**< Total EASGD communication time */
    int easgd_sync_count;           /**< Number of EASGD synchronizations */
} Stage1Worker;

/**
 * @brief Stage2 Worker structure for hybrid system
 */
typedef struct {
    /* Worker identification */
    int worker_id;                  /**< Always 1 for single Stage2 worker */
    
    /* Pipeline components */
    NetworkStage* stage2;           /**< Local Stage2 network: 512 → 10 */
    
    /* EASGD parameters */
    double alpha2;                  /**< Elastic averaging rate for Stage2 */
    double* stage2_weights_flat;    /**< Flattened Stage2 weights for EASGD */
    int stage2_weight_count;        /**< Number of Stage2 parameters */
    
    /* Multiplexed pipeline communication */
    int stage1_1_forward_server;    /**< Receive from Stage1-1 */
    int stage1_2_forward_server;    /**< Receive from Stage1-2 */
    int stage1_1_backward_client;   /**< Send gradients to Stage1-1 */
    int stage1_2_backward_client;   /**< Send gradients to Stage1-2 */
    
    /* Client sockets for active connections */
    int stage1_1_forward_client;    /**< Active connection from Stage1-1 */
    int stage1_2_forward_client;    /**< Active connection from Stage1-2 */
    
    /* Message multiplexing */
    MessageQueue stage1_1_queue;    /**< Message queue from Stage1-1 */
    MessageQueue stage1_2_queue;    /**< Message queue from Stage1-2 */
    
    /* EASGD communication */
    int last_easgd_socket;          /**< Short-lived EASGD socket per epoch */
    
    /* Performance metrics */
    PipelineStats local_stats;      /**< Local pipeline statistics */
    double total_easgd_comm_time;   /**< Total EASGD communication time */
    int messages_from_stage1_1;     /**< Messages processed from Stage1-1 */
    int messages_from_stage1_2;     /**< Messages processed from Stage1-2 */
    
    /* Training control */
    int* training_active;           /**< Shared training status flag */
} Stage2Worker;

/* =============================================================================
 * HYBRID COORDINATOR STRUCTURE
 * ============================================================================= */

/**
 * @brief Hybrid Coordinator + Parameter Server structure
 */
typedef struct {
    /* EASGD Parameter Server components */
    double beta;                    /**< Server update rate (β parameter) */
    
    /* Master weights */
    double* master_stage1_weights;  /**< w̄₁: Central Stage1 weights */
    double* master_stage2_weights;  /**< w̄₂: Central Stage2 weights */
    int stage1_weight_count;        /**< Number of Stage1 parameters */
    int stage2_weight_count;        /**< Number of Stage2 parameters */
    pthread_mutex_t weights_mutex;  /**< Mutex for thread-safe weight updates */
    
    /* Worker management */
    int num_stage1_workers;         /**< Number of Stage1 workers (2) */
    int num_stage2_workers;         /**< Number of Stage2 workers (1) */
    int current_epoch;              /**< Current training epoch */
    int communication_round;        /**< Current EASGD round */
    
    /* Communication servers */
    int easgd_server_socket;        /**< EASGD communication server */
    int stats_server_socket;        /**< Statistics collection server */
    
    /* Performance monitoring (inherited + extended) */
    PipelineStats aggregated_stats; /**< Aggregated pipeline statistics */
    
    /* EASGD-specific metrics */
    double total_weight_divergence; /**< Total weight divergence */
    double stage1_divergence;       /**< Stage1 weight divergence */
    double stage2_divergence;       /**< Stage2 weight divergence */
    double easgd_communication_time; /**< Total EASGD communication time */
    int total_easgd_rounds;         /**< Total EASGD communication rounds */
    
    /* Hybrid system metrics */
    double pipeline_efficiency;     /**< Pipeline utilization efficiency */
    double easgd_efficiency;        /**< EASGD synchronization efficiency */
    double hybrid_speedup;          /**< Speedup vs sequential baseline */
    
    /* Thread safety */
    pthread_mutex_t stats_mutex;    /**< Statistics protection */
    
    /* Timing */
    time_t start_time;              /**< Training start time */
    time_t last_easgd_sync;         /**< Last EASGD synchronization time */
    
    /* Training control */
    int* training_active;           /**< Global training status flag */
} HybridCoordinator;

/* =============================================================================
 * HYBRID EXPERIMENT RESULTS
 * ============================================================================= */

/**
 * @brief Extended experiment results for hybrid system
 */
typedef struct {
    /* Base experiment info (inherited) */
    ExperimentResults base_results; /**< Original experiment results */
    
    /* EASGD-specific results */
    double alpha1_parameter;        /**< α₁ hyperparameter used */
    double alpha2_parameter;        /**< α₂ hyperparameter used */
    double beta_parameter;          /**< β hyperparameter used */
    
    /* Convergence analysis */
    double final_weight_divergence; /**< Final weight divergence */
    double convergence_rate_stage1; /**< Stage1 convergence rate */
    double convergence_rate_stage2; /**< Stage2 convergence rate */
    int convergence_epoch;          /**< Epoch where convergence achieved */
    
    /* Performance comparison */
    double pipeline_only_time;      /**< Estimated pipeline-only time */
    double easgd_only_time;         /**< Estimated EASGD-only time */
    double hybrid_speedup_factor;   /**< Hybrid vs individual approaches */
    
    /* Load balancing analysis */
    double stage1_1_utilization;    /**< Stage1-1 utilization percentage */
    double stage1_2_utilization;    /**< Stage1-2 utilization percentage */
    double stage2_utilization;      /**< Stage2-1 utilization percentage */
    double load_balance_variance;   /**< Load balance variance */
    
    /* Communication analysis */
    double pipeline_comm_ratio;     /**< Pipeline communication overhead */
    double easgd_comm_ratio;        /**< EASGD communication overhead */
    double total_network_usage;     /**< Total network bandwidth used */
} HybridExperimentResults;

/* =============================================================================
 * FUNCTION DECLARATIONS - EASGD CORE
 * ============================================================================= */

/* EASGD message handling */
EASGDMessage* create_easgd_message(EASGDMessageType type, int worker_id, int stage_type);
void free_easgd_message(EASGDMessage* msg);
int send_easgd_message(int sockfd, EASGDMessage* msg);
EASGDMessage* receive_easgd_message(int sockfd);

/* Weight flattening and reconstruction */
int flatten_stage1_weights(Stage1Worker* worker);
int flatten_stage2_weights(Stage2Worker* worker);
int reconstruct_stage1_weights(Stage1Worker* worker, double* flat_weights);
int reconstruct_stage2_weights(Stage2Worker* worker, double* flat_weights);

/* EASGD algorithm implementation */
int update_master_stage1_weights(HybridCoordinator* coord, double* worker1_weights, 
                                double* worker2_weights);
int update_master_stage2_weights(HybridCoordinator* coord, double* worker_weights);
int apply_elastic_averaging_stage1(Stage1Worker* worker, double* master_weights);
int apply_elastic_averaging_stage2(Stage2Worker* worker, double* master_weights);

/* =============================================================================
 * FUNCTION DECLARATIONS - HYBRID WORKERS
 * ============================================================================= */

/* Stage1 Worker functions */
Stage1Worker* create_stage1_worker(int worker_id, int data_start, int data_end, double alpha1);
void hybrid_stage1_main(int worker_id, int data_start, int data_end, double alpha1);
int send_stage1_weights_to_server(Stage1Worker* worker, int epoch);
int receive_master_stage1_weights(Stage1Worker* worker);
void cleanup_stage1_worker(Stage1Worker* worker);

/* Stage2 Worker functions */
Stage2Worker* create_stage2_worker(double alpha2);
void hybrid_stage2_main(double alpha2);
int process_multiplexed_messages(Stage2Worker* worker);
int send_stage2_weights_to_server(Stage2Worker* worker, int epoch);
int receive_master_stage2_weights(Stage2Worker* worker);
void cleanup_stage2_worker(Stage2Worker* worker);

/* Async thread functions for hybrid workers */
void* hybrid_stage1_forward_processor(void* args);
void* hybrid_stage1_backward_processor(void* args);

/* Message queue functions */
int init_message_queue(MessageQueue* queue);
int enqueue_message(MessageQueue* queue, ForwardMessage* msg);
ForwardMessage* dequeue_message(MessageQueue* queue);
void cleanup_message_queue(MessageQueue* queue);

/* =============================================================================
 * FUNCTION DECLARATIONS - HYBRID COORDINATOR
 * ============================================================================= */

/* Coordinator creation and management */
HybridCoordinator* create_hybrid_coordinator(double beta);
void hybrid_coordinator_main(double beta);
void cleanup_hybrid_coordinator(HybridCoordinator* coord);

/* EASGD coordination functions */
int collect_worker_weights(HybridCoordinator* coord, int epoch);
int distribute_master_weights(HybridCoordinator* coord, int epoch);
int perform_easgd_update(HybridCoordinator* coord, EASGDMessage** stage1_msgs, 
                        EASGDMessage* stage2_msg);

/* Master weight management */
int send_master_stage1_weights(HybridCoordinator* coord, int client_socket, int worker_id);
int send_master_stage2_weights(HybridCoordinator* coord, int client_socket);
int initialize_master_weights(HybridCoordinator* coord);

/* =============================================================================
 * FUNCTION DECLARATIONS - HYBRID METRICS & ANALYSIS
 * ============================================================================= */

/* Divergence calculation */
double calculate_stage1_weight_divergence(HybridCoordinator* coord, 
                                        EASGDMessage* worker1_msg, EASGDMessage* worker2_msg);
double calculate_stage2_weight_divergence(HybridCoordinator* coord, EASGDMessage* worker_msg);
double calculate_total_system_divergence(HybridCoordinator* coord);

/* Performance analysis */
HybridExperimentResults* analyze_hybrid_performance(HybridCoordinator* coord);
void generate_hybrid_research_report(HybridExperimentResults* results);
int save_hybrid_metrics_to_csv(const char* filename, HybridExperimentResults* results);

/* Load balancing analysis */
double calculate_load_balance_variance(HybridCoordinator* coord);
void optimize_load_distribution(HybridCoordinator* coord);

/* Convergence analysis */
int detect_convergence(HybridCoordinator* coord, double threshold);
void track_convergence_metrics(HybridCoordinator* coord, int epoch);

/* =============================================================================
 * FUNCTION DECLARATIONS - UTILITY FUNCTIONS
 * ============================================================================= */

/* Configuration helpers */
void set_optimal_hybrid_parameters(double* alpha1, double* alpha2, double* beta, 
                                  const char* network_type);
int validate_hybrid_configuration(int num_stage1_workers, int num_stage2_workers);

/* Dataset partitioning */
Img** load_dataset_partition(int start_idx, int end_idx, const char* dataset_path);

/* Debugging and monitoring */
void log_easgd_synchronization(int round, double divergence, double comm_time);
void log_hybrid_performance(HybridCoordinator* coord, int epoch);
void debug_weight_consistency(HybridCoordinator* coord);

#endif /* HYBRID_PIPELINE_EASGD_H */