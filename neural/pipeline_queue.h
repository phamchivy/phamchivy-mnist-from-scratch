// pipeline_queue.h - UNIFIED ASYNC CONTEXT
#ifndef PIPELINE_QUEUE_H
#define PIPELINE_QUEUE_H

#include "../matrix/matrix.h"
#include <pthread.h>
#include <stdbool.h>
#include <sys/mman.h>
#include <fcntl.h>
#include "../neural/pipeline_utils.h"
#include "../config/config_loader.h"

// Forward declarations
// typedef struct PipelineStage PipelineStage;
// typedef struct HybridConfig HybridConfig;

// Configuration for queue sizes
#define DEFAULT_QUEUE_SIZE 5
#define MAX_QUEUE_SIZE 20

// Activation message between Stage 1 → Stage 2
typedef struct {
    Matrix* activation;          // Hidden layer output
    int label;                  // True label for this sample
    long long timestamp;        // For debugging/profiling
    int batch_id;              // Batch identifier
} ActivationMessage;

// Gradient message between Stage 2 → Stage 1  
typedef struct {
    Matrix* gradient;           // Gradient for Stage 1
    long long timestamp;        // For debugging/profiling
    int batch_id;              // Corresponding batch ID
    double loss;               // Loss value for this sample
} GradientMessage;

// Thread-safe activation queue (Stage 1 → Stage 2)
typedef struct {
    ActivationMessage* messages;
    int head;                   // Next position to dequeue
    int tail;                   // Next position to enqueue
    int size;                   // Current number of items
    int capacity;               // Maximum capacity
    
    // Thread synchronization
    pthread_mutex_t mutex;
    pthread_cond_t not_empty;   // Signal when queue has items
    pthread_cond_t not_full;    // Signal when queue has space
    
    // Statistics
    long long total_enqueued;
    long long total_dequeued;
    bool shutdown;              // Graceful shutdown flag
} ActivationQueue;

// Thread-safe gradient queue (Stage 2 → Stage 1)
typedef struct {
    GradientMessage* messages;
    int head;
    int tail; 
    int size;
    int capacity;
    
    // Thread synchronization
    pthread_mutex_t mutex;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
    
    // Statistics
    long long total_enqueued;
    long long total_dequeued;
    bool shutdown;
} GradientQueue;

// ✅ SHARED MEMORY STATE (moved from shared_pipeline.h)
typedef struct {
    volatile bool stage1_ready;
    volatile bool stage2_ready;
    volatile bool shutdown_requested;
    
    // Statistics shared between stages
    long long total_activations_sent;
    long long total_gradients_sent;
    double avg_processing_latency_ms;
    
    // Backpressure control
    volatile bool stage2_busy;
    volatile int pending_activations;
    
    // Padding to avoid false sharing
    char padding[64];
} SharedPipelineState;

// ✅ UNIFIED ASYNC PIPELINE CONTEXT
typedef struct {
    // Core queue components
    ActivationQueue* activation_queue;
    GradientQueue* gradient_queue;
    
    // Thread handles
    pthread_t stage1_thread;
    pthread_t stage2_thread;
    
    // Pipeline stages
    PipelineStage* stage1;
    PipelineStage* stage2;
    
    // Configuration
    int queue_size;
    int group_id;
    bool enable_profiling;
    HybridConfig* config;       // Configuration reference
    
    // Statistics
    double avg_queue_wait_time;
    double total_processing_time;
    long long samples_processed;
    
    // ✅ SHARED MEMORY COMPONENTS (optional, only used if enabled)
    SharedPipelineState* shared_state;
    char shared_memory_name[64];
    int shared_fd;
    bool use_shared_memory;     // Flag to indicate if shared memory is used
    
    // Connection management
    bool connected;
    long long connection_timestamp;
} AsyncPipelineContext;

// ===== BASIC QUEUE FUNCTIONS =====
ActivationQueue* activation_queue_create(int capacity);
void activation_queue_destroy(ActivationQueue* queue);
bool activation_queue_enqueue(ActivationQueue* queue, Matrix* activation, int label, int batch_id);
bool activation_queue_dequeue(ActivationQueue* queue, ActivationMessage* msg, int timeout_ms);
void activation_queue_shutdown(ActivationQueue* queue);

GradientQueue* gradient_queue_create(int capacity);
void gradient_queue_destroy(GradientQueue* queue);
bool gradient_queue_enqueue(GradientQueue* queue, Matrix* gradient, int batch_id, double loss);
bool gradient_queue_dequeue(GradientQueue* queue, GradientMessage* msg, int timeout_ms);
void gradient_queue_shutdown(GradientQueue* queue);

// ===== ASYNC PIPELINE FUNCTIONS =====
// Basic async pipeline (in-memory queues only)
AsyncPipelineContext* async_pipeline_create(int queue_size, int group_id);
void async_pipeline_destroy(AsyncPipelineContext* ctx);

// Shared memory async pipeline
AsyncPipelineContext* async_pipeline_create_shared(int queue_size, int group_id, const char* shared_key);
void async_pipeline_destroy_shared(AsyncPipelineContext* ctx);

// Pipeline control
int async_pipeline_start(AsyncPipelineContext* ctx, PipelineStage* stage1, PipelineStage* stage2);
void async_pipeline_stop(AsyncPipelineContext* ctx);

// ===== SHARED MEMORY FUNCTIONS (only available for shared contexts) =====
void async_pipeline_set_stage1_ready(AsyncPipelineContext* ctx);
void async_pipeline_set_stage2_ready(AsyncPipelineContext* ctx);
bool async_pipeline_is_stage1_ready(AsyncPipelineContext* ctx);
bool async_pipeline_is_stage2_ready(AsyncPipelineContext* ctx);

// Enhanced queue operations with timeout
bool activation_queue_enqueue_timeout(ActivationQueue* queue, Matrix* activation, 
                                     int label, int batch_id, int timeout_ms);

// Backpressure management
void async_pipeline_update_backpressure(AsyncPipelineContext* ctx);
bool async_pipeline_check_backpressure(AsyncPipelineContext* ctx);

// ===== UTILITY FUNCTIONS =====
long long get_timestamp_us(void);  // Microsecond timestamp
void print_queue_stats(ActivationQueue* aq, GradientQueue* gq);

// Thread functions (implemented in worker files)
void* async_stage1_worker(void* arg);
void* async_stage2_worker(void* arg);

#endif