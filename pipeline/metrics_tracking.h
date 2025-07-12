#ifndef METRICS_TRACKING_H
#define METRICS_TRACKING_H

#include <sys/time.h>
#include "pipeline_nn.h"

// Metrics tracking function declarations
void track_pipeline_bubble(const char* reason);
void track_synchronization_time(struct timeval start, struct timeval end);
void track_idle_time(struct timeval start, struct timeval end);
void log_real_time_metrics(const char* stage_name, int epoch, int batch, PipelineStats* stats);
int setup_stage_connection(const char* stage_name, const char* target_ip, int port, int retries);

// Global metrics access functions
int get_total_pipeline_bubbles(void);
double get_total_sync_time(void);
double get_total_idle_time(void);
void reset_metrics_counters(void);

// Metrics summary functions
void print_metrics_summary(const char* stage_name, PipelineStats* stats, double total_time);
void log_epoch_summary(const char* stage_name, int epoch, PipelineStats* stats);

#endif // METRICS_TRACKING_H 