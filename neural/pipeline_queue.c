// pipeline_queue.c - UNIFIED IMPLEMENTATION
#include "pipeline_queue.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <errno.h>
#include <unistd.h>

// Get current timestamp in microseconds
long long get_timestamp_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000LL + tv.tv_usec;
}

// ===== ACTIVATION QUEUE IMPLEMENTATION =====

ActivationQueue* activation_queue_create(int capacity) {
    if (capacity <= 0 || capacity > MAX_QUEUE_SIZE) {
        capacity = DEFAULT_QUEUE_SIZE;
    }
    
    ActivationQueue* queue = malloc(sizeof(ActivationQueue));
    if (!queue) return NULL;
    
    queue->messages = malloc(sizeof(ActivationMessage) * capacity);
    if (!queue->messages) {
        free(queue);
        return NULL;
    }
    
    queue->head = 0;
    queue->tail = 0;
    queue->size = 0;
    queue->capacity = capacity;
    queue->total_enqueued = 0;
    queue->total_dequeued = 0;
    queue->shutdown = false;
    
    // Initialize mutex and conditions
    if (pthread_mutex_init(&queue->mutex, NULL) != 0) {
        free(queue->messages);
        free(queue);
        return NULL;
    }
    
    if (pthread_cond_init(&queue->not_empty, NULL) != 0 ||
        pthread_cond_init(&queue->not_full, NULL) != 0) {
        pthread_mutex_destroy(&queue->mutex);
        free(queue->messages);
        free(queue);
        return NULL;
    }
    
    return queue;
}

void activation_queue_destroy(ActivationQueue* queue) {
    if (!queue) return;
    
    pthread_mutex_lock(&queue->mutex);
    
    // Free any remaining messages
    for (int i = 0; i < queue->size; i++) {
        int idx = (queue->head + i) % queue->capacity;
        matrix_free(queue->messages[idx].activation);
    }
    
    pthread_mutex_unlock(&queue->mutex);
    
    pthread_mutex_destroy(&queue->mutex);
    pthread_cond_destroy(&queue->not_empty);
    pthread_cond_destroy(&queue->not_full);
    
    free(queue->messages);
    free(queue);
}

bool activation_queue_enqueue(ActivationQueue* queue, Matrix* activation, int label, int batch_id) {
    if (!queue || !activation) return false;
    
    pthread_mutex_lock(&queue->mutex);
    
    // Wait for space if queue is full
    while (queue->size == queue->capacity && !queue->shutdown) {
        pthread_cond_wait(&queue->not_full, &queue->mutex);
    }
    
    if (queue->shutdown) {
        pthread_mutex_unlock(&queue->mutex);
        return false;
    }
    
    // Add message to queue
    ActivationMessage* msg = &queue->messages[queue->tail];
    msg->activation = matrix_copy(activation);  // Deep copy for thread safety
    msg->label = label;
    msg->batch_id = batch_id;
    msg->timestamp = get_timestamp_us();
    
    queue->tail = (queue->tail + 1) % queue->capacity;
    queue->size++;
    queue->total_enqueued++;
    
    // Signal waiting dequeue operations
    pthread_cond_signal(&queue->not_empty);
    pthread_mutex_unlock(&queue->mutex);
    
    return true;
}

bool activation_queue_dequeue(ActivationQueue* queue, ActivationMessage* msg, int timeout_ms) {
    if (!queue || !msg) return false;
    
    pthread_mutex_lock(&queue->mutex);
    
    // Setup timeout
    struct timespec abs_timeout;
    if (timeout_ms > 0) {
        clock_gettime(CLOCK_REALTIME, &abs_timeout);
        abs_timeout.tv_sec += timeout_ms / 1000;
        abs_timeout.tv_nsec += (timeout_ms % 1000) * 1000000;
        if (abs_timeout.tv_nsec >= 1000000000) {
            abs_timeout.tv_sec++;
            abs_timeout.tv_nsec -= 1000000000;
        }
    }
    
    // Wait for items if queue is empty
    while (queue->size == 0 && !queue->shutdown) {
        if (timeout_ms > 0) {
            int result = pthread_cond_timedwait(&queue->not_empty, &queue->mutex, &abs_timeout);
            if (result == ETIMEDOUT) {
                pthread_mutex_unlock(&queue->mutex);
                return false;
            }
        } else {
            pthread_cond_wait(&queue->not_empty, &queue->mutex);
        }
    }
    
    if (queue->shutdown && queue->size == 0) {
        pthread_mutex_unlock(&queue->mutex);
        return false;
    }
    
    // Remove message from queue
    *msg = queue->messages[queue->head];
    queue->head = (queue->head + 1) % queue->capacity;
    queue->size--;
    queue->total_dequeued++;
    
    // Signal waiting enqueue operations
    pthread_cond_signal(&queue->not_full);
    pthread_mutex_unlock(&queue->mutex);
    
    return true;
}

void activation_queue_shutdown(ActivationQueue* queue) {
    if (!queue) return;
    
    pthread_mutex_lock(&queue->mutex);
    queue->shutdown = true;
    pthread_cond_broadcast(&queue->not_empty);
    pthread_cond_broadcast(&queue->not_full);
    pthread_mutex_unlock(&queue->mutex);
}

// ===== GRADIENT QUEUE IMPLEMENTATION =====

GradientQueue* gradient_queue_create(int capacity) {
    if (capacity <= 0 || capacity > MAX_QUEUE_SIZE) {
        capacity = DEFAULT_QUEUE_SIZE;
    }
    
    GradientQueue* queue = malloc(sizeof(GradientQueue));
    if (!queue) return NULL;
    
    queue->messages = malloc(sizeof(GradientMessage) * capacity);
    if (!queue->messages) {
        free(queue);
        return NULL;
    }
    
    queue->head = 0;
    queue->tail = 0;
    queue->size = 0;
    queue->capacity = capacity;
    queue->total_enqueued = 0;
    queue->total_dequeued = 0;
    queue->shutdown = false;
    
    // Initialize mutex and conditions
    if (pthread_mutex_init(&queue->mutex, NULL) != 0) {
        free(queue->messages);
        free(queue);
        return NULL;
    }
    
    if (pthread_cond_init(&queue->not_empty, NULL) != 0 ||
        pthread_cond_init(&queue->not_full, NULL) != 0) {
        pthread_mutex_destroy(&queue->mutex);
        free(queue->messages);
        free(queue);
        return NULL;
    }
    
    return queue;
}

void gradient_queue_destroy(GradientQueue* queue) {
    if (!queue) return;
    
    pthread_mutex_lock(&queue->mutex);
    
    // Free any remaining messages
    for (int i = 0; i < queue->size; i++) {
        int idx = (queue->head + i) % queue->capacity;
        matrix_free(queue->messages[idx].gradient);
    }
    
    pthread_mutex_unlock(&queue->mutex);
    
    pthread_mutex_destroy(&queue->mutex);
    pthread_cond_destroy(&queue->not_empty);
    pthread_cond_destroy(&queue->not_full);
    
    free(queue->messages);
    free(queue);
}

bool gradient_queue_enqueue(GradientQueue* queue, Matrix* gradient, int batch_id, double loss) {
    if (!queue || !gradient) return false;
    
    pthread_mutex_lock(&queue->mutex);
    
    while (queue->size == queue->capacity && !queue->shutdown) {
        pthread_cond_wait(&queue->not_full, &queue->mutex);
    }
    
    if (queue->shutdown) {
        pthread_mutex_unlock(&queue->mutex);
        return false;
    }
    
    GradientMessage* msg = &queue->messages[queue->tail];
    msg->gradient = matrix_copy(gradient);
    msg->batch_id = batch_id;
    msg->loss = loss;
    msg->timestamp = get_timestamp_us();
    
    queue->tail = (queue->tail + 1) % queue->capacity;
    queue->size++;
    queue->total_enqueued++;
    
    pthread_cond_signal(&queue->not_empty);
    pthread_mutex_unlock(&queue->mutex);
    
    return true;
}

bool gradient_queue_dequeue(GradientQueue* queue, GradientMessage* msg, int timeout_ms) {
    if (!queue || !msg) return false;
    
    pthread_mutex_lock(&queue->mutex);
    
    struct timespec abs_timeout;
    if (timeout_ms > 0) {
        clock_gettime(CLOCK_REALTIME, &abs_timeout);
        abs_timeout.tv_sec += timeout_ms / 1000;
        abs_timeout.tv_nsec += (timeout_ms % 1000) * 1000000;
        if (abs_timeout.tv_nsec >= 1000000000) {
            abs_timeout.tv_sec++;
            abs_timeout.tv_nsec -= 1000000000;
        }
    }
    
    while (queue->size == 0 && !queue->shutdown) {
        if (timeout_ms > 0) {
            int result = pthread_cond_timedwait(&queue->not_empty, &queue->mutex, &abs_timeout);
            if (result == ETIMEDOUT) {
                pthread_mutex_unlock(&queue->mutex);
                return false;
            }
        } else {
            pthread_cond_wait(&queue->not_empty, &queue->mutex);
        }
    }
    
    if (queue->shutdown && queue->size == 0) {
        pthread_mutex_unlock(&queue->mutex);
        return false;
    }
    
    *msg = queue->messages[queue->head];
    queue->head = (queue->head + 1) % queue->capacity;
    queue->size--;
    queue->total_dequeued++;
    
    pthread_cond_signal(&queue->not_full);
    pthread_mutex_unlock(&queue->mutex);
    
    return true;
}

void gradient_queue_shutdown(GradientQueue* queue) {
    if (!queue) return;
    
    pthread_mutex_lock(&queue->mutex);
    queue->shutdown = true;
    pthread_cond_broadcast(&queue->not_empty);
    pthread_cond_broadcast(&queue->not_full);
    pthread_mutex_unlock(&queue->mutex);
}

// ===== ASYNC PIPELINE CONTEXT IMPLEMENTATION =====

// ✅ Basic async pipeline (in-memory only)
AsyncPipelineContext* async_pipeline_create(int queue_size, int group_id) {
    AsyncPipelineContext* ctx = malloc(sizeof(AsyncPipelineContext));
    if (!ctx) return NULL;
    
    ctx->activation_queue = activation_queue_create(queue_size);
    ctx->gradient_queue = gradient_queue_create(queue_size);
    
    if (!ctx->activation_queue || !ctx->gradient_queue) {
        if (ctx->activation_queue) activation_queue_destroy(ctx->activation_queue);
        if (ctx->gradient_queue) gradient_queue_destroy(ctx->gradient_queue);
        free(ctx);
        return NULL;
    }
    
    // Initialize all fields
    ctx->stage1 = NULL;
    ctx->stage2 = NULL;
    ctx->queue_size = queue_size;
    ctx->group_id = group_id;
    ctx->enable_profiling = false;
    ctx->config = NULL;
    ctx->avg_queue_wait_time = 0.0;
    ctx->total_processing_time = 0.0;
    ctx->samples_processed = 0;
    
    // ✅ No shared memory components
    ctx->shared_state = NULL;
    memset(ctx->shared_memory_name, 0, sizeof(ctx->shared_memory_name));
    ctx->shared_fd = -1;
    ctx->use_shared_memory = false;
    
    ctx->connected = false;
    ctx->connection_timestamp = get_timestamp_us();
    
    return ctx;
}

// ✅ Shared memory async pipeline
// ✅ FIX trong pipeline_queue.c

AsyncPipelineContext* async_pipeline_create_shared(int queue_size, int group_id, const char* shared_key) {
    printf("[Shared Memory] Creating shared pipeline for group %d with key: %s\n", group_id, shared_key);
    
    AsyncPipelineContext* ctx = async_pipeline_create(queue_size, group_id);
    if (!ctx) {
        printf("[Shared Memory] Failed to create basic async pipeline context\n");
        return NULL;
    }
    
    // Initialize shared memory components
    strncpy(ctx->shared_memory_name, shared_key, sizeof(ctx->shared_memory_name) - 1);
    ctx->shared_memory_name[sizeof(ctx->shared_memory_name) - 1] = '\0';
    ctx->use_shared_memory = true;
    
    printf("[Shared Memory] Attempting to create/open shared memory: %s\n", ctx->shared_memory_name);
    
    // ✅ FIX 1: Try to connect first (for Stage 2)
    ctx->shared_fd = shm_open(ctx->shared_memory_name, O_RDWR, 0666);
    if (ctx->shared_fd != -1) {
        printf("[Shared Memory] Connected to existing shared memory\n");
    } else {
        printf("[Shared Memory] Existing shared memory not found, creating new one\n");
        
        // Clean up any stale shared memory first
        shm_unlink(ctx->shared_memory_name);
        
        // Create new shared memory segment
        ctx->shared_fd = shm_open(ctx->shared_memory_name, O_CREAT | O_RDWR, 0666);
        if (ctx->shared_fd == -1) {
            printf("[Shared Memory] Failed to create shared memory '%s': %s\n", 
                   ctx->shared_memory_name, strerror(errno));
            
            // Fallback to in-memory mode
            ctx->use_shared_memory = false;
            ctx->shared_state = NULL;
            ctx->shared_fd = -1;
            printf("[Shared Memory] Falling back to in-memory mode\n");
            return ctx;
        }
        
        // ✅ FIX 2: Only set size if we created it
        if (ftruncate(ctx->shared_fd, sizeof(SharedPipelineState)) == -1) {
            printf("[Shared Memory] Failed to set shared memory size: %s\n", strerror(errno));
            close(ctx->shared_fd);
            shm_unlink(ctx->shared_memory_name);
            // Fall back to in-memory
            ctx->use_shared_memory = false;
            ctx->shared_state = NULL;
            ctx->shared_fd = -1;
            return ctx;
        }
        
        printf("[Shared Memory] Created new shared memory, size: %zu bytes\n", sizeof(SharedPipelineState));
    }
    
    printf("[Shared Memory] Shared memory ready, fd=%d\n", ctx->shared_fd);
    
    // Map shared memory (same for both create and connect)
    ctx->shared_state = mmap(NULL, sizeof(SharedPipelineState), 
                            PROT_READ | PROT_WRITE, MAP_SHARED, ctx->shared_fd, 0);
    if (ctx->shared_state == MAP_FAILED) {
        printf("[Shared Memory] Failed to map shared memory: %s\n", strerror(errno));
        close(ctx->shared_fd);
        shm_unlink(ctx->shared_memory_name);
        // Fall back to in-memory
        ctx->use_shared_memory = false;
        ctx->shared_state = NULL;
        ctx->shared_fd = -1;
        return ctx;
    }
    
    printf("[Shared Memory] Memory mapped successfully at %p\n", ctx->shared_state);
    
    // ✅ FIX 3: Only initialize if we're the first process (creator)
    // Check if shared state is already initialized
    if (ctx->shared_state->stage1_ready == false && ctx->shared_state->stage2_ready == false) {
        // We're the first process, initialize
        memset(ctx->shared_state, 0, sizeof(SharedPipelineState));
        printf("[Shared Memory] Initialized shared state (first process)\n");
    } else {
        printf("[Shared Memory] Connected to existing initialized shared state\n");
    }
    
    printf("[Shared Pipeline Group %d] Created with shared memory key: %s (mode: shared)\n", 
           group_id, shared_key);
    
    return ctx;
}

void async_pipeline_destroy(AsyncPipelineContext* ctx) {
    if (!ctx) return;
    
    if (ctx->activation_queue) activation_queue_destroy(ctx->activation_queue);
    if (ctx->gradient_queue) gradient_queue_destroy(ctx->gradient_queue);
    
    free(ctx);
}

void async_pipeline_destroy_shared(AsyncPipelineContext* ctx) {
    if (!ctx) return;
    
    // Cleanup shared memory
    if (ctx->use_shared_memory) {
        if (ctx->shared_state && ctx->shared_state != MAP_FAILED) {
            munmap(ctx->shared_state, sizeof(SharedPipelineState));
        }
        
        if (ctx->shared_fd >= 0) {
            close(ctx->shared_fd);
            shm_unlink(ctx->shared_memory_name); // Only last process will succeed
        }
    }
    
    // Use regular destroy for the rest
    async_pipeline_destroy(ctx);
}

// ===== SHARED MEMORY FUNCTIONS =====

void async_pipeline_set_stage1_ready(AsyncPipelineContext* ctx) {
    if (ctx && ctx->use_shared_memory && ctx->shared_state) {
        ctx->shared_state->stage1_ready = true;
        printf("[Shared Pipeline Group %d] Stage1 marked as ready\n", ctx->group_id);
    }
}

void async_pipeline_set_stage2_ready(AsyncPipelineContext* ctx) {
    if (ctx && ctx->use_shared_memory && ctx->shared_state) {
        ctx->shared_state->stage2_ready = true;
        printf("[Shared Pipeline Group %d] Stage2 marked as ready\n", ctx->group_id);
    }
}

bool async_pipeline_is_stage1_ready(AsyncPipelineContext* ctx) {
    if (ctx && ctx->use_shared_memory && ctx->shared_state) {
        return ctx->shared_state->stage1_ready;
    }
    return false;
}

bool async_pipeline_is_stage2_ready(AsyncPipelineContext* ctx) {
    if (ctx && ctx->use_shared_memory && ctx->shared_state) {
        return ctx->shared_state->stage2_ready;
    }
    return false;
}

// Enhanced queue operations with timeout
bool activation_queue_enqueue_timeout(ActivationQueue* queue, Matrix* activation, 
                                     int label, int batch_id, int timeout_ms) {
    if (!queue || !activation) return false;
    
    struct timespec start, current;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    pthread_mutex_lock(&queue->mutex);
    
    while (queue->size == queue->capacity && !queue->shutdown) {
        // Check timeout
        if (timeout_ms > 0) {
            clock_gettime(CLOCK_MONOTONIC, &current);
            long long elapsed_ms = ((current.tv_sec - start.tv_sec) * 1000) + 
                                  ((current.tv_nsec - start.tv_nsec) / 1000000);
            
            if (elapsed_ms >= timeout_ms) {
                pthread_mutex_unlock(&queue->mutex);
                return false; // Timeout
            }
            
            // Wait with remaining timeout
            struct timespec abs_timeout;
            clock_gettime(CLOCK_REALTIME, &abs_timeout);
            abs_timeout.tv_sec += (timeout_ms - elapsed_ms) / 1000;
            abs_timeout.tv_nsec += ((timeout_ms - elapsed_ms) % 1000) * 1000000;
            
            if (abs_timeout.tv_nsec >= 1000000000) {
                abs_timeout.tv_sec++;
                abs_timeout.tv_nsec -= 1000000000;
            }
            
            int result = pthread_cond_timedwait(&queue->not_full, &queue->mutex, &abs_timeout);
            if (result == ETIMEDOUT) {
                pthread_mutex_unlock(&queue->mutex);
                return false;
            }
        } else {
            pthread_cond_wait(&queue->not_full, &queue->mutex);
        }
    }
    
    if (queue->shutdown) {
        pthread_mutex_unlock(&queue->mutex);
        return false;
    }
    
    // Add message to queue
    ActivationMessage* msg = &queue->messages[queue->tail];
    msg->activation = matrix_copy(activation);
    msg->label = label;
    msg->batch_id = batch_id;
    msg->timestamp = get_timestamp_us();
    
    queue->tail = (queue->tail + 1) % queue->capacity;
    queue->size++;
    queue->total_enqueued++;
    
    pthread_cond_signal(&queue->not_empty);
    pthread_mutex_unlock(&queue->mutex);
    
    return true;
}

// Backpressure management
void async_pipeline_update_backpressure(AsyncPipelineContext* ctx) {
    if (!ctx || !ctx->use_shared_memory || !ctx->shared_state) return;
    
    pthread_mutex_lock(&ctx->activation_queue->mutex);
    int pending = ctx->activation_queue->size;
    pthread_mutex_unlock(&ctx->activation_queue->mutex);
    
    ctx->shared_state->pending_activations = pending;
    ctx->shared_state->stage2_busy = (pending > (ctx->queue_size * 0.8)); // 80% threshold
}

bool async_pipeline_check_backpressure(AsyncPipelineContext* ctx) {
    if (!ctx || !ctx->use_shared_memory || !ctx->shared_state) return false;
    return ctx->shared_state->stage2_busy;
}

// ===== UTILITY FUNCTIONS =====

void print_queue_stats(ActivationQueue* aq, GradientQueue* gq) {
    printf("=== Pipeline Queue Statistics ===\n");
    
    if (aq) {
        pthread_mutex_lock(&aq->mutex);
        printf("Activation Queue: size=%d/%d, enqueued=%lld, dequeued=%lld\n",
               aq->size, aq->capacity, aq->total_enqueued, aq->total_dequeued);
        pthread_mutex_unlock(&aq->mutex);
    }
    
    if (gq) {
        pthread_mutex_lock(&gq->mutex);
        printf("Gradient Queue: size=%d/%d, enqueued=%lld, dequeued=%lld\n",
               gq->size, gq->capacity, gq->total_enqueued, gq->total_dequeued);
        pthread_mutex_unlock(&gq->mutex);
    }
    
    printf("================================\n");
}

// Basic pipeline control (implementations depend on specific worker needs)
int async_pipeline_start(AsyncPipelineContext* ctx, PipelineStage* stage1, PipelineStage* stage2) {
    if (!ctx) return -1;
    
    ctx->stage1 = stage1;
    ctx->stage2 = stage2;
    
    printf("[Async Pipeline Group %d] Started with queue size %d\n", 
           ctx->group_id, ctx->queue_size);
    
    return 0;
}

void async_pipeline_stop(AsyncPipelineContext* ctx) {
    if (!ctx) return;
    
    // Signal shutdown
    activation_queue_shutdown(ctx->activation_queue);
    gradient_queue_shutdown(ctx->gradient_queue);
    
    // Print final stats
    print_queue_stats(ctx->activation_queue, ctx->gradient_queue);
    
    printf("[Async Pipeline Group %d] Stopped after processing %lld samples\n",
           ctx->group_id, ctx->samples_processed);
}