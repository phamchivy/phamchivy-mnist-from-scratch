// batch_storage.h
#ifndef BATCH_STORAGE_H
#define BATCH_STORAGE_H

#include "../matrix/matrix.h"
#include <pthread.h>
#include <stdbool.h>

#define MAX_STORED_BATCHES 100

// Stored batch data for gradient matching
typedef struct {
    int batch_id;
    Matrix* input;
    Matrix* hidden_output;
    long long timestamp;
    bool used;
} StoredBatch;

// Thread-safe batch storage
typedef struct {
    StoredBatch batches[MAX_STORED_BATCHES];
    int next_slot;
    int stored_count;
    pthread_mutex_t mutex;
    
    // Statistics
    long long total_stored;
    long long total_retrieved;
    long long cleanup_count;
} BatchStorage;

// Function declarations
BatchStorage* batch_storage_create(void);
void batch_storage_destroy(BatchStorage* storage);
bool batch_storage_store(BatchStorage* storage, int batch_id, Matrix* input, Matrix* hidden_output);
bool batch_storage_retrieve(BatchStorage* storage, int batch_id, Matrix** input, Matrix** hidden_output);
void batch_storage_cleanup_old(BatchStorage* storage, long long max_age_us);
void batch_storage_print_stats(BatchStorage* storage);

#endif