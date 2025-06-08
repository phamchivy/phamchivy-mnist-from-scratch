#include "pipeline_nn.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int initialize_buffer(PipelineBuffer* buffer) {
    if (!buffer) return -1;
    
    buffer->forward_head = 0;
    buffer->forward_tail = 0;
    buffer->backward_head = 0;
    buffer->backward_tail = 0;
    
    // Initialize all queue pointers to NULL
    for (int i = 0; i < MAX_PIPELINE_DEPTH; i++) {
        buffer->forward_queue[i] = NULL;
        buffer->backward_queue[i] = NULL;
    }
    
    // Initialize mutexes and condition variables
    if (pthread_mutex_init(&buffer->forward_mutex, NULL) != 0) {
        return -1;
    }
    if (pthread_mutex_init(&buffer->backward_mutex, NULL) != 0) {
        pthread_mutex_destroy(&buffer->forward_mutex);
        return -1;
    }
    if (pthread_cond_init(&buffer->forward_cond, NULL) != 0) {
        pthread_mutex_destroy(&buffer->forward_mutex);
        pthread_mutex_destroy(&buffer->backward_mutex);
        return -1;
    }
    if (pthread_cond_init(&buffer->backward_cond, NULL) != 0) {
        pthread_mutex_destroy(&buffer->forward_mutex);
        pthread_mutex_destroy(&buffer->backward_mutex);
        pthread_cond_destroy(&buffer->forward_cond);
        return -1;
    }
    
    return 0;
}

int enqueue_forward(PipelineBuffer* buffer, ForwardMessage* msg) {
    if (!buffer || !msg) return -1;
    
    pthread_mutex_lock(&buffer->forward_mutex);
    
    int next_tail = (buffer->forward_tail + 1) % MAX_PIPELINE_DEPTH;
    
    // Check if buffer is full
    if (next_tail == buffer->forward_head) {
        pthread_mutex_unlock(&buffer->forward_mutex);
        return -1; // Buffer full
    }
    
    buffer->forward_queue[buffer->forward_tail] = msg;
    buffer->forward_tail = next_tail;
    
    // Signal waiting threads
    pthread_cond_signal(&buffer->forward_cond);
    pthread_mutex_unlock(&buffer->forward_mutex);
    
    return 0;
}

ForwardMessage* dequeue_forward(PipelineBuffer* buffer) {
    if (!buffer) return NULL;
    
    pthread_mutex_lock(&buffer->forward_mutex);
    
    // Wait while buffer is empty
    while (buffer->forward_head == buffer->forward_tail) {
        pthread_cond_wait(&buffer->forward_cond, &buffer->forward_mutex);
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
    
    // Check if buffer is full
    if (next_tail == buffer->backward_head) {
        pthread_mutex_unlock(&buffer->backward_mutex);
        return -1; // Buffer full
    }
    
    buffer->backward_queue[buffer->backward_tail] = msg;
    buffer->backward_tail = next_tail;
    
    // Signal waiting threads
    pthread_cond_signal(&buffer->backward_cond);
    pthread_mutex_unlock(&buffer->backward_mutex);
    
    return 0;
}

BackwardMessage* dequeue_backward(PipelineBuffer* buffer) {
    if (!buffer) return NULL;
    
    pthread_mutex_lock(&buffer->backward_mutex);
    
    // Wait while buffer is empty
    while (buffer->backward_head == buffer->backward_tail) {
        pthread_cond_wait(&buffer->backward_cond, &buffer->backward_mutex);
    }
    
    BackwardMessage* msg = buffer->backward_queue[buffer->backward_head];
    buffer->backward_queue[buffer->backward_head] = NULL;
    buffer->backward_head = (buffer->backward_head + 1) % MAX_PIPELINE_DEPTH;
    
    pthread_mutex_unlock(&buffer->backward_mutex);
    
    return msg;
}

void cleanup_buffer(PipelineBuffer* buffer) {
    if (!buffer) return;
    
    // Clean up any remaining messages
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
    
    // Destroy synchronization objects
    pthread_mutex_destroy(&buffer->forward_mutex);
    pthread_mutex_destroy(&buffer->backward_mutex);
    pthread_cond_destroy(&buffer->forward_cond);
    pthread_cond_destroy(&buffer->backward_cond);
} 