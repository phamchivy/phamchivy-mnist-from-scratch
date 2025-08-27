// batch_storage.c
#include "batch_storage.h"
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <limits.h>

static long long get_timestamp_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000LL + tv.tv_usec;
}

BatchStorage* batch_storage_create(void) {
    BatchStorage* storage = malloc(sizeof(BatchStorage));
    if (!storage) return NULL;
    
    // Initialize all batches as unused
    for (int i = 0; i < MAX_STORED_BATCHES; i++) {
        storage->batches[i].used = false;
        storage->batches[i].input = NULL;
        storage->batches[i].hidden_output = NULL;
    }
    
    storage->next_slot = 0;
    storage->stored_count = 0;
    storage->total_stored = 0;
    storage->total_retrieved = 0;
    storage->cleanup_count = 0;
    
    if (pthread_mutex_init(&storage->mutex, NULL) != 0) {
        free(storage);
        return NULL;
    }
    
    return storage;
}

void batch_storage_destroy(BatchStorage* storage) {
    if (!storage) return;
    
    pthread_mutex_lock(&storage->mutex);
    
    // Free all stored matrices
    for (int i = 0; i < MAX_STORED_BATCHES; i++) {
        if (storage->batches[i].used) {
            if (storage->batches[i].input) matrix_free(storage->batches[i].input);
            if (storage->batches[i].hidden_output) matrix_free(storage->batches[i].hidden_output);
        }
    }
    
    pthread_mutex_unlock(&storage->mutex);
    pthread_mutex_destroy(&storage->mutex);
    free(storage);
}

bool batch_storage_store(BatchStorage* storage, int batch_id, Matrix* input, Matrix* hidden_output) {
    if (!storage || !input || !hidden_output) return false;
    
    pthread_mutex_lock(&storage->mutex);
    
    // Find a slot (circular buffer with replacement)
    int slot = storage->next_slot;
    
    // Free existing data if slot is occupied
    if (storage->batches[slot].used) {
        matrix_free(storage->batches[slot].input);
        matrix_free(storage->batches[slot].hidden_output);
        storage->stored_count--;
    }
    
    // Store new batch (deep copy for thread safety)
    storage->batches[slot].batch_id = batch_id;
    storage->batches[slot].input = matrix_copy(input);
    storage->batches[slot].hidden_output = matrix_copy(hidden_output);
    storage->batches[slot].timestamp = get_timestamp_us();
    storage->batches[slot].used = true;
    
    storage->next_slot = (storage->next_slot + 1) % MAX_STORED_BATCHES;
    storage->stored_count++;
    storage->total_stored++;
    
    pthread_mutex_unlock(&storage->mutex);
    return true;
}

bool batch_storage_retrieve(BatchStorage* storage, int batch_id, Matrix** input, Matrix** hidden_output) {
    if (!storage || !input || !hidden_output) return false;
    
    pthread_mutex_lock(&storage->mutex);
    
    // Search for batch_id
    for (int i = 0; i < MAX_STORED_BATCHES; i++) {
        if (storage->batches[i].used && storage->batches[i].batch_id == batch_id) {
            // Found matching batch
            *input = storage->batches[i].input;
            *hidden_output = storage->batches[i].hidden_output;
            
            // Mark as unused (don't free here, caller will free)
            storage->batches[i].used = false;
            storage->batches[i].input = NULL;
            storage->batches[i].hidden_output = NULL;
            
            storage->stored_count--;
            storage->total_retrieved++;
            
            pthread_mutex_unlock(&storage->mutex);
            return true;
        }
    }
    
    pthread_mutex_unlock(&storage->mutex);
    return false; // Not found
}

void batch_storage_cleanup_old(BatchStorage* storage, long long max_age_us) {
    if (!storage) return;
    
    pthread_mutex_lock(&storage->mutex);
    
    long long current_time = get_timestamp_us();
    int cleaned = 0;
    
    for (int i = 0; i < MAX_STORED_BATCHES; i++) {
        if (storage->batches[i].used && 
            (current_time - storage->batches[i].timestamp) > max_age_us) {
            
            matrix_free(storage->batches[i].input);
            matrix_free(storage->batches[i].hidden_output);
            storage->batches[i].used = false;
            storage->batches[i].input = NULL;
            storage->batches[i].hidden_output = NULL;
            
            storage->stored_count--;
            cleaned++;
        }
    }
    
    storage->cleanup_count += cleaned;
    pthread_mutex_unlock(&storage->mutex);
    
    if (cleaned > 0) {
        printf("[BatchStorage] Cleaned up %d old batches\n", cleaned);
    }
}

// ✅ ADD: Missing print_stats function
void batch_storage_print_stats(BatchStorage* storage) {
    if (!storage) return;
    
    pthread_mutex_lock(&storage->mutex);
    
    printf("=== Batch Storage Statistics ===\n");
    printf("Current stored: %d/%d (%.1f%% full)\n", 
           storage->stored_count, MAX_STORED_BATCHES,
           (double)storage->stored_count / MAX_STORED_BATCHES * 100);
    printf("Total stored: %lld\n", storage->total_stored);
    printf("Total retrieved: %lld\n", storage->total_retrieved);
    printf("Total cleaned up: %lld\n", storage->cleanup_count);
    printf("Hit rate: %.2f%% (%lld/%lld)\n",
           storage->total_stored > 0 ? 
           (double)storage->total_retrieved / storage->total_stored * 100 : 0.0,
           storage->total_retrieved, storage->total_stored);
    
    // Show age distribution of current batches
    if (storage->stored_count > 0) {
        long long current_time = get_timestamp_us();
        long long oldest_age = 0, newest_age = LLONG_MAX;
        int age_buckets[5] = {0}; // <1s, 1-5s, 5-10s, 10-30s, >30s
        
        for (int i = 0; i < MAX_STORED_BATCHES; i++) {
            if (storage->batches[i].used) {
                long long age = current_time - storage->batches[i].timestamp;
                
                if (age > oldest_age) oldest_age = age;
                if (age < newest_age) newest_age = age;
                
                // Categorize by age
                if (age < 1000000) age_buckets[0]++;        // <1s
                else if (age < 5000000) age_buckets[1]++;   // 1-5s
                else if (age < 10000000) age_buckets[2]++;  // 5-10s
                else if (age < 30000000) age_buckets[3]++;  // 10-30s
                else age_buckets[4]++;                      // >30s
            }
        }
        
        printf("Batch ages: newest=%.1fs, oldest=%.1fs\n",
               newest_age / 1000000.0, oldest_age / 1000000.0);
        printf("Age distribution: <1s:%d, 1-5s:%d, 5-10s:%d, 10-30s:%d, >30s:%d\n",
               age_buckets[0], age_buckets[1], age_buckets[2], age_buckets[3], age_buckets[4]);
    }
    
    printf("===============================\n");
    
    pthread_mutex_unlock(&storage->mutex);
}