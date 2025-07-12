#include "metrics_tracking.h"
#include "../socket/socket_utils.h"
#include <stdio.h>
#include <unistd.h>
#include <time.h>

// External global variables from pipeline_main.c
extern double total_idle_time;
extern double total_sync_time;
extern int total_pipeline_bubbles;

// External utility function from pipeline_main.c
extern double get_time_diff(struct timeval start, struct timeval end);

// Enhanced connection setup with metrics tracking
int setup_stage_connection(const char* stage_name, const char* target_ip, int port, int retries) {
    struct timeval conn_start, conn_end;
    gettimeofday(&conn_start, NULL);
    
    int sockfd = -1;
    for (int retry = 0; retry < retries; retry++) {
        if (retry > 0) {
            printf("[%s] Connection attempt %d/%d to %s:%d\n", stage_name, retry + 1, retries, target_ip, port);
            sleep(1);
        }
        
        sockfd = connect_to_server(target_ip, port);
        if (sockfd >= 0) {
            gettimeofday(&conn_end, NULL);
            double conn_time = get_time_diff(conn_start, conn_end);
            printf("[%s] ✅ Connected to %s:%d in %.3f seconds\n", stage_name, target_ip, port, conn_time);
            
            // Track connection as synchronization time
            track_synchronization_time(conn_start, conn_end);
            return sockfd;
        }
    }
    
    gettimeofday(&conn_end, NULL);
    double failed_time = get_time_diff(conn_start, conn_end);
    printf("[%s] ❌ Failed to connect to %s:%d after %.3f seconds\n", stage_name, target_ip, port, failed_time);
    
    // Track failed connection as idle time
    track_idle_time(conn_start, conn_end);
    return -1;
}

// Pipeline bubble tracking
void track_pipeline_bubble(const char* reason) {
    total_pipeline_bubbles++;
    printf("[METRICS] 🔄 Pipeline bubble #%d: %s\n", total_pipeline_bubbles, reason);
}

// Synchronization time tracking
void track_synchronization_time(struct timeval start, struct timeval end) {
    double sync_time = get_time_diff(start, end);
    total_sync_time += sync_time;
    printf("[METRICS] ⏱️  Sync time: +%.3f seconds (total: %.3f)\n", sync_time, total_sync_time);
}

// Idle time tracking
void track_idle_time(struct timeval start, struct timeval end) {
    double idle_time = get_time_diff(start, end);
    total_idle_time += idle_time;
    printf("[METRICS] 💤 Idle time: +%.3f seconds (total: %.3f)\n", idle_time, total_idle_time);
}

// Real-time metrics logging
void log_real_time_metrics(const char* stage_name, int epoch, int batch, PipelineStats* stats) {
    printf("\n[%s] 📊 REAL-TIME METRICS (Epoch %d, Batch %d):\n", stage_name, epoch, batch);
    printf("├── Processing Time: Stage1=%.3fs, Stage2=%.3fs\n", 
           stats->stage1_processing_time, stats->stage2_processing_time);
    printf("├── Communication Time: %.3fs (%.1f%% overhead)\n", 
           stats->communication_time, stats->communication_overhead);
    printf("├── Pipeline Bubbles: %d\n", total_pipeline_bubbles);
    printf("├── Sync Time: %.3fs, Idle Time: %.3fs\n", total_sync_time, total_idle_time);
    printf("├── Throughput: %.2f samples/sec\n", stats->samples_per_second);
    printf("├── Load Balance: %.2f (ideal: 1.0)\n", stats->load_balance_ratio);
    printf("└── Accuracy: %.2f%% (Loss: %.4f)\n", 
           stats->total_predictions > 0 ? (double)stats->correct_predictions / stats->total_predictions * 100.0 : 0.0,
           stats->processed_batches > 0 ? stats->total_loss / stats->processed_batches : 0.0);
    printf("\n");
    fflush(stdout);
}

// Global metrics access functions
int get_total_pipeline_bubbles(void) {
    return total_pipeline_bubbles;
}

double get_total_sync_time(void) {
    return total_sync_time;
}

double get_total_idle_time(void) {
    return total_idle_time;
}

void reset_metrics_counters(void) {
    total_pipeline_bubbles = 0;
    total_sync_time = 0.0;
    total_idle_time = 0.0;
    printf("[METRICS] 🔄 Metrics counters reset\n");
}

// Enhanced metrics summary
void print_metrics_summary(const char* stage_name, PipelineStats* stats, double total_time) {
    printf("\n[%s] 🎉 FINAL METRICS SUMMARY:\n", stage_name);
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│ ⏱️  TIME BREAKDOWN                                           │\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│ Total time: %.2f seconds                                   │\n", total_time);
    printf("│ Processing: %.3fs (%.1f%%)                                 │\n", 
           stats->stage1_processing_time + stats->stage2_processing_time,
           (stats->stage1_processing_time + stats->stage2_processing_time) / total_time * 100.0);
    printf("│ Communication: %.3fs (%.1f%%)                              │\n", 
           stats->communication_time, stats->communication_time / total_time * 100.0);
    printf("│ Synchronization: %.3fs (%.1f%%)                            │\n", 
           total_sync_time, total_sync_time / total_time * 100.0);
    printf("│ Idle time: %.3fs (%.1f%%)                                  │\n", 
           total_idle_time, total_idle_time / total_time * 100.0);
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│ 📊 PERFORMANCE METRICS                                      │\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│ Pipeline bubbles: %d                                        │\n", total_pipeline_bubbles);
    printf("│ Pipeline efficiency: %.2f%%                                 │\n", stats->pipeline_efficiency);
    printf("│ Throughput: %.2f samples/second                             │\n", stats->samples_per_second);
    printf("│ Load balance ratio: %.2f                                    │\n", stats->load_balance_ratio);
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│ 🎯 TRAINING RESULTS                                         │\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│ Batches processed: %d                                       │\n", stats->processed_batches);
    printf("│ Final accuracy: %.2f%%                                      │\n", 
           stats->total_predictions > 0 ? (double)stats->correct_predictions / stats->total_predictions * 100.0 : 0.0);
    printf("│ Final loss: %.4f                                            │\n", 
           stats->processed_batches > 0 ? stats->total_loss / stats->processed_batches : 0.0);
    printf("└─────────────────────────────────────────────────────────────┘\n");
    fflush(stdout);
}

// Epoch summary logging
void log_epoch_summary(const char* stage_name, int epoch, PipelineStats* stats) {
    printf("[%s] 🏁 Epoch %d Summary:\n", stage_name, epoch);
    printf("├── Batches: %d\n", stats->processed_batches);
    printf("├── Pipeline bubbles: %d\n", total_pipeline_bubbles);
    printf("├── Accuracy: %.2f%%\n", 
           stats->total_predictions > 0 ? (double)stats->correct_predictions / stats->total_predictions * 100.0 : 0.0);
    printf("├── Loss: %.4f\n", 
           stats->processed_batches > 0 ? stats->total_loss / stats->processed_batches : 0.0);
    printf("├── Sync time: %.3fs, Idle time: %.3fs\n", total_sync_time, total_idle_time);
    printf("└── Throughput: %.2f samples/sec\n", stats->samples_per_second);
    fflush(stdout);
} 