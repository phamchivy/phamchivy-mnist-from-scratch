/**
 * @file buffer.c
 * @brief Pipeline buffer implementation for asynchronous message passing
 * @author Research Team
 * @date 2024
 * @version 1.0
 * 
 * This file implements thread-safe circular buffer queues for managing
 * forward and backward messages in the pipeline parallelism system.
 * It provides asynchronous message passing capabilities with retry mechanisms
 * for handling failed transmissions.
 * 
 * Key features:
 * - Thread-safe circular buffer implementation
 * - Forward and backward message queues
 * - Retry mechanism for failed message transmissions
 * - Timeout-based queue operations
 * - Comprehensive error handling and recovery
 */

#include "pipeline_nn.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <errno.h>

int initialize_buffer(PipelineBuffer* buffer) {
    if (!buffer) return -1;
    
    buffer->forward_head = 0;
    buffer->forward_tail = 0;
    buffer->backward_head = 0;
    buffer->backward_tail = 0;
    buffer->retry_head = 0;
    buffer->retry_tail = 0;
    buffer->retry_count = 0;
    
    /* Initialize all queue pointers to NULL */
    for (int i = 0; i < MAX_PIPELINE_DEPTH; i++) {
        buffer->forward_queue[i] = NULL;
        buffer->backward_queue[i] = NULL;
        buffer->retry_queue[i] = NULL;
    }
    
    /* Initialize mutex and condition variables */
    if (pthread_mutex_init(&buffer->forward_mutex, NULL) != 0) {
        return -1;
    }
    if (pthread_mutex_init(&buffer->backward_mutex, NULL) != 0) {
        pthread_mutex_destroy(&buffer->forward_mutex);
        return -1;
    }
    if (pthread_mutex_init(&buffer->retry_mutex, NULL) != 0) {
        pthread_mutex_destroy(&buffer->forward_mutex);
        pthread_mutex_destroy(&buffer->backward_mutex);
        return -1;
    }
    if (pthread_cond_init(&buffer->forward_cond, NULL) != 0) {
        pthread_mutex_destroy(&buffer->forward_mutex);
        pthread_mutex_destroy(&buffer->backward_mutex);
        pthread_mutex_destroy(&buffer->retry_mutex);
        return -1;
    }
    if (pthread_cond_init(&buffer->backward_cond, NULL) != 0) {
        pthread_mutex_destroy(&buffer->forward_mutex);
        pthread_mutex_destroy(&buffer->backward_mutex);
        pthread_mutex_destroy(&buffer->retry_mutex);
        pthread_cond_destroy(&buffer->forward_cond);
        return -1;
    }
    if (pthread_cond_init(&buffer->retry_cond, NULL) != 0) {
        pthread_mutex_destroy(&buffer->forward_mutex);
        pthread_mutex_destroy(&buffer->backward_mutex);
        pthread_mutex_destroy(&buffer->retry_mutex);
        pthread_cond_destroy(&buffer->forward_cond);
        pthread_cond_destroy(&buffer->backward_cond);
        return -1;
    }
    
    return 0;
}

int enqueue_forward(PipelineBuffer* buffer, ForwardMessage* msg) {
    if (!buffer || !msg) return -1;
    
    pthread_mutex_lock(&buffer->forward_mutex);
    
    int next_tail = (buffer->forward_tail + 1) % MAX_PIPELINE_DEPTH;
    
    /* Check if buffer is full */
    if (next_tail == buffer->forward_head) {
        pthread_mutex_unlock(&buffer->forward_mutex);
        return -1; /* Buffer full */
    }
    
    buffer->forward_queue[buffer->forward_tail] = msg;
    buffer->forward_tail = next_tail;
    
    /* Signal waiting threads */
    pthread_cond_signal(&buffer->forward_cond);
    pthread_mutex_unlock(&buffer->forward_mutex);
    
    return 0;
}

ForwardMessage* dequeue_forward(PipelineBuffer* buffer) {
    if (!buffer) return NULL;
    
    pthread_mutex_lock(&buffer->forward_mutex);
    
    /* Wait with timeout when buffer is empty */
    struct timespec timeout;
    clock_gettime(CLOCK_REALTIME, &timeout);
    timeout.tv_sec += 1; /* 1 second timeout */
    
    while (buffer->forward_head == buffer->forward_tail) {
        int wait_result = pthread_cond_timedwait(&buffer->forward_cond, &buffer->forward_mutex, &timeout);
        if (wait_result == ETIMEDOUT) {
            pthread_mutex_unlock(&buffer->forward_mutex);
            return NULL; /* Timeout - return NULL so thread can check training_active */
        }
        if (wait_result != 0) {
            pthread_mutex_unlock(&buffer->forward_mutex);
            return NULL; /* Other error */
        }
    }
    
    ForwardMessage* msg = buffer->forward_queue[buffer->forward_head];
    buffer->forward_queue[buffer->forward_head] = NULL;
    buffer->forward_head = (buffer->forward_head + 1) % MAX_PIPELINE_DEPTH;
    
    pthread_mutex_unlock(&buffer->forward_mutex);
    
    return msg;
}

int enqueue_backward(PipelineBuffer* buffer, BackwardMessage* msg) {
    if (!buffer || !msg) return -1;
    
    pthread_mutex_lock(&buffer->backward_mutex);
    
    int next_tail = (buffer->backward_tail + 1) % MAX_PIPELINE_DEPTH;
    
    /* Check if buffer is full */
    if (next_tail == buffer->backward_head) {
        pthread_mutex_unlock(&buffer->backward_mutex);
        return -1; /* Buffer full */
    }
    
    buffer->backward_queue[buffer->backward_tail] = msg;
    buffer->backward_tail = next_tail;
    
    /* Signal waiting threads */
    pthread_cond_signal(&buffer->backward_cond);
    pthread_mutex_unlock(&buffer->backward_mutex);
    
    return 0;
}

BackwardMessage* dequeue_backward(PipelineBuffer* buffer) {
    if (!buffer) return NULL;
    
    pthread_mutex_lock(&buffer->backward_mutex);
    
    /* Wait when buffer is empty */
    while (buffer->backward_head == buffer->backward_tail) {
        pthread_cond_wait(&buffer->backward_cond, &buffer->backward_mutex);
    }
    
    BackwardMessage* msg = buffer->backward_queue[buffer->backward_head];
    buffer->backward_queue[buffer->backward_head] = NULL;
    buffer->backward_head = (buffer->backward_head + 1) % MAX_PIPELINE_DEPTH;
    
    pthread_mutex_unlock(&buffer->backward_mutex);
    
    return msg;
}

/* =============================================================================
 * RETRY QUEUE FUNCTIONS
 * ============================================================================= */

/**
 * @brief Enqueue message to retry queue
 * @param buffer Pointer to pipeline buffer
 * @param msg Pointer to forward message
 * @return 0 on success, -1 on failure
 */
int enqueue_retry(PipelineBuffer* buffer, ForwardMessage* msg) {
    if (!buffer || !msg) return -1;
    
    pthread_mutex_lock(&buffer->retry_mutex);
    
    int next_tail = (buffer->retry_tail + 1) % MAX_PIPELINE_DEPTH;
    
    /* Check if retry queue is full */
    if (next_tail == buffer->retry_head) {
        pthread_mutex_unlock(&buffer->retry_mutex);
        return -1; /* Retry queue full */
    }
    
    buffer->retry_queue[buffer->retry_tail] = msg;
    buffer->retry_tail = next_tail;
    buffer->retry_count++;
    
    /* Signal waiting threads */
    pthread_cond_signal(&buffer->retry_cond);
    pthread_mutex_unlock(&buffer->retry_mutex);
    
    return 0;
}

ForwardMessage* dequeue_retry(PipelineBuffer* buffer) {
    if (!buffer) return NULL;
    
    pthread_mutex_lock(&buffer->retry_mutex);
    
    /* Check if retry queue is empty */
    if (buffer->retry_head == buffer->retry_tail) {
        pthread_mutex_unlock(&buffer->retry_mutex);
        return NULL;
    }
    
    ForwardMessage* msg = buffer->retry_queue[buffer->retry_head];
    buffer->retry_queue[buffer->retry_head] = NULL;
    buffer->retry_head = (buffer->retry_head + 1) % MAX_PIPELINE_DEPTH;
    buffer->retry_count--;
    
    pthread_mutex_unlock(&buffer->retry_mutex);
    
    return msg;
}

int get_retry_count(PipelineBuffer* buffer) {
    if (!buffer) return 0;
    
    pthread_mutex_lock(&buffer->retry_mutex);
    int count = buffer->retry_count;
    pthread_mutex_unlock(&buffer->retry_mutex);
    
    return count;
}

void process_retry_queue(PipelineBuffer* buffer) {
    if (!buffer) return;
    
    /* Attempt to move messages from retry queue to main forward queue */
    ForwardMessage* msg;
    while ((msg = dequeue_retry(buffer)) != NULL) {
        if (enqueue_forward(buffer, msg) < 0) {
            /* Still cannot enqueue, put back in retry queue */
            enqueue_retry(buffer, msg);
            break;
        }
    }
}

void cleanup_buffer(PipelineBuffer* buffer) {
    if (!buffer) return;
    
    /* Clean up remaining messages */
    pthread_mutex_lock(&buffer->forward_mutex);
    while (buffer->forward_head != buffer->forward_tail) {
        ForwardMessage* msg = buffer->forward_queue[buffer->forward_head];
        if (msg) {
            free_forward_message(msg);
        }
        buffer->forward_head = (buffer->forward_head + 1) % MAX_PIPELINE_DEPTH;
    }
    pthread_mutex_unlock(&buffer->forward_mutex);
    
    pthread_mutex_lock(&buffer->backward_mutex);
    while (buffer->backward_head != buffer->backward_tail) {
        BackwardMessage* msg = buffer->backward_queue[buffer->backward_head];
        if (msg) {
            free_backward_message(msg);
        }
        buffer->backward_head = (buffer->backward_head + 1) % MAX_PIPELINE_DEPTH;
    }
    pthread_mutex_unlock(&buffer->backward_mutex);
    
    /* Clean up retry queue */
    pthread_mutex_lock(&buffer->retry_mutex);
    while (buffer->retry_head != buffer->retry_tail) {
        ForwardMessage* msg = buffer->retry_queue[buffer->retry_head];
        if (msg) {
            free_forward_message(msg);
        }
        buffer->retry_head = (buffer->retry_head + 1) % MAX_PIPELINE_DEPTH;
    }
    pthread_mutex_unlock(&buffer->retry_mutex);
    
    /* Destroy synchronization objects */
    pthread_mutex_destroy(&buffer->forward_mutex);
    pthread_mutex_destroy(&buffer->backward_mutex);
    pthread_mutex_destroy(&buffer->retry_mutex);
    pthread_cond_destroy(&buffer->forward_cond);
    pthread_cond_destroy(&buffer->backward_cond);
    pthread_cond_destroy(&buffer->retry_cond);
}

/* =============================================================================
 * GLOBAL VARIABLES FOR METRICS COLLECTION
 * ============================================================================= */

/* Note: These variables are defined in pipeline_main.c */
/* extern int total_out_of_order_messages; */
/* extern int batch_counter; */
/* extern int current_epoch; */

/* =============================================================================
 * HELPER FUNCTIONS FOR QUEUE SIZE CALCULATION
 * ============================================================================= */
int get_forward_queue_size(PipelineBuffer* buffer) {
    if (!buffer) return 0;
    
    pthread_mutex_lock(&buffer->forward_mutex);
    int size = (buffer->forward_tail - buffer->forward_head + MAX_PIPELINE_DEPTH) % MAX_PIPELINE_DEPTH;
    pthread_mutex_unlock(&buffer->forward_mutex);
    
    return size;
}

int get_backward_queue_size(PipelineBuffer* buffer) {
    if (!buffer) return 0;
    
    pthread_mutex_lock(&buffer->backward_mutex);
    int size = (buffer->backward_tail - buffer->backward_head + MAX_PIPELINE_DEPTH) % MAX_PIPELINE_DEPTH;
    pthread_mutex_unlock(&buffer->backward_mutex);
    
    return size;
}

/* =============================================================================
 * METRICS LOGGING FUNCTIONS
 * ============================================================================= */
void log_queue_sizes(PipelineBuffer* buffer, int batch_count, int epoch) {
    if (!buffer) return;
    
    int forward_size = get_forward_queue_size(buffer);
    int backward_size = get_backward_queue_size(buffer);
    long timestamp = get_current_timestamp();
    
    printf("METRIC_LOG|%ld|QUEUE_SIZE|%d|%d|%d|%d\n", timestamp, forward_size, backward_size, batch_count, epoch);
    fflush(stdout);
}

void log_bubble_rate(int total_bubbles, int batch_count, int epoch) {
    if (batch_count == 0) return;
    
    double bubble_rate = (double)total_bubbles / batch_count;
    long timestamp = get_current_timestamp();
    
    printf("METRIC_LOG|%ld|BUBBLE_RATE|%.4f|%d|%d|%d\n", timestamp, bubble_rate, total_bubbles, batch_count, epoch);
    fflush(stdout);
}

void log_out_of_order_message(int batch_id, int expected_id, int total_out_of_order, int epoch) {
    long timestamp = get_current_timestamp();
    
    printf("METRIC_LOG|%ld|OUT_ORDER|%d|%d|%d|%d\n", timestamp, batch_id, expected_id, total_out_of_order, epoch);
    fflush(stdout);
} 