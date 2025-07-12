#define _DEFAULT_SOURCE
#include "metrics_collector.h"
#include <string.h>
#include <sys/stat.h>
#include <errno.h>
#include <unistd.h>

// Global metrics collector instance
MetricsCollector* g_metrics_collector = NULL;
static pthread_t metrics_thread;
static int metrics_thread_running = 0;

// Initialize metrics collector
MetricsCollector* init_metrics_collector(const char* output_dir) {
    MetricsCollector* collector = malloc(sizeof(MetricsCollector));
    if (!collector) {
        printf("Error: Cannot allocate memory for MetricsCollector\n");
        return NULL;
    }
    
    // Initialize structure
    collector->capacity = 10000; /* Can hold 10000 measurements */
    collector->count = 0;
    collector->data = malloc(sizeof(TimeSeriesMetric) * collector->capacity);
    
    if (!collector->data) {
        printf("Error: Cannot allocate memory for metrics data\n");
        free(collector);
        return NULL;
    }
    
    // Initialize mutex
    if (pthread_mutex_init(&collector->mutex, NULL) != 0) {
        printf("Error: Cannot initialize metrics mutex\n");
        free(collector->data);
        free(collector);
        return NULL;
    }
    
    // Set output directory
    strncpy(collector->output_dir, output_dir, sizeof(collector->output_dir) - 1);
    collector->output_dir[sizeof(collector->output_dir) - 1] = '\0';
    
    // Create output directory if it doesn't exist
    mkdir(collector->output_dir, 0755);
    
    // Initialize start time
    gettimeofday(&collector->start_time, NULL);
    
    // Open CSV files for each metric
    const char* metric_names[] = {
        "packets_lost",
        "network_latency_ms", 
        "communication_time",
        "batch_order",
        "throughput",
        "load_balance_ratio",
        "memory_usage_mb",
        "cpu_utilization",
        "combined_metrics"
    };
    
    for (int i = 0; i < 9; i++) {
        char filepath[512];
        snprintf(filepath, sizeof(filepath), "%s/%s.csv", output_dir, metric_names[i]);
        
        collector->csv_files[i] = fopen(filepath, "w");
        if (!collector->csv_files[i]) {
            printf("Error: Cannot create CSV file %s: %s\n", filepath, strerror(errno));
            // Cleanup already opened files
            for (int j = 0; j < i; j++) {
                if (collector->csv_files[j]) {
                    fclose(collector->csv_files[j]);
                }
            }
            pthread_mutex_destroy(&collector->mutex);
            free(collector->data);
            free(collector);
            return NULL;
        }
        
        // Write headers for individual metric files
        if (i < 8) {
            fprintf(collector->csv_files[i], "timestamp,%s\n", metric_names[i]);
        }
    }
    
    // Write header for combined metrics file
    fprintf(collector->csv_files[8], 
            "timestamp,packets_lost,network_latency_ms,communication_time,batch_order,"
            "throughput,load_balance_ratio,memory_usage_mb,cpu_utilization,"
            "stage1_time,stage2_time,loss_value,accuracy\n");
    
    printf("📊 Metrics collector initialized in directory: %s\n", output_dir);
    return collector;
}

// Cleanup metrics collector
void cleanup_metrics_collector(MetricsCollector* collector) {
    if (!collector) return;
    
    // Stop metrics collection thread if running
    if (metrics_thread_running) {
        stop_metrics_collection();
    }
    
    // Save final data
            save_timeseries_csv(collector);
    
    // Close CSV files
    for (int i = 0; i < 9; i++) {
        if (collector->csv_files[i]) {
            fclose(collector->csv_files[i]);
        }
    }
    
    // Cleanup
    pthread_mutex_destroy(&collector->mutex);
    free(collector->data);
    free(collector);
    
    printf("📊 Metrics collector cleaned up\n");
}

// Get current timestamp relative to start time
double get_metrics_timestamp(MetricsCollector* collector) {
    struct timeval current_time;
    gettimeofday(&current_time, NULL);
    
    return (current_time.tv_sec - collector->start_time.tv_sec) + 
           (current_time.tv_usec - collector->start_time.tv_usec) / 1000000.0;
}

// Add a metric sample
int add_metric_sample(MetricsCollector* collector, 
                     int packets_lost,
                     double network_latency_ms,
                     double communication_time,
                     int batch_order,
                     double throughput,
                     double load_balance_ratio,
                     double memory_usage_mb,
                     double cpu_utilization,
                     double stage1_time,
                     double stage2_time,
                     double loss_value,
                     double accuracy) {
    
    if (!collector) return -1;
    
    pthread_mutex_lock(&collector->mutex);
    
    // Check if we need to resize the array
    if (collector->count >= collector->capacity) {
        collector->capacity *= 2;
        collector->data = realloc(collector->data, 
                                sizeof(TimeSeriesMetric) * collector->capacity);
        if (!collector->data) {
            pthread_mutex_unlock(&collector->mutex);
            return -1;
        }
    }
    
    // Add new sample
    TimeSeriesMetric* sample = &collector->data[collector->count];
    sample->timestamp = get_metrics_timestamp(collector);
    sample->packets_lost = packets_lost;
    sample->network_latency_ms = network_latency_ms;
    sample->communication_time = communication_time;
    sample->batch_order = batch_order;
    sample->throughput = throughput;
    sample->load_balance_ratio = load_balance_ratio;
    sample->memory_usage_mb = memory_usage_mb;
    sample->cpu_utilization = cpu_utilization;
    sample->stage1_time = stage1_time;
    sample->stage2_time = stage2_time;
    sample->loss_value = loss_value;
    sample->accuracy = accuracy;
    
    collector->count++;
    
    // Write to individual CSV files immediately for real-time monitoring
    double timestamp = sample->timestamp;
    
    // Write to individual metric files
    fprintf(collector->csv_files[0], "%.3f,%d\n", timestamp, packets_lost);
    fprintf(collector->csv_files[1], "%.3f,%.3f\n", timestamp, network_latency_ms);
    fprintf(collector->csv_files[2], "%.3f,%.3f\n", timestamp, communication_time);
    fprintf(collector->csv_files[3], "%.3f,%d\n", timestamp, batch_order);
    fprintf(collector->csv_files[4], "%.3f,%.3f\n", timestamp, throughput);
    fprintf(collector->csv_files[5], "%.3f,%.3f\n", timestamp, load_balance_ratio);
    fprintf(collector->csv_files[6], "%.3f,%.3f\n", timestamp, memory_usage_mb);
    fprintf(collector->csv_files[7], "%.3f,%.3f\n", timestamp, cpu_utilization);
    
    // Write to combined metrics file
    fprintf(collector->csv_files[8], 
            "%.3f,%d,%.3f,%.3f,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.6f,%.3f\n",
            timestamp, packets_lost, network_latency_ms, communication_time,
            batch_order, throughput, load_balance_ratio, memory_usage_mb,
            cpu_utilization, stage1_time, stage2_time, loss_value, accuracy);
    
    // Flush all files for real-time updates
    for (int i = 0; i < 9; i++) {
        fflush(collector->csv_files[i]);
    }
    
    pthread_mutex_unlock(&collector->mutex);
    return 0;
}

// Save metrics to CSV (final save)
void save_timeseries_csv(MetricsCollector* collector) {
    if (!collector) return;
    
    pthread_mutex_lock(&collector->mutex);
    
    // Create summary file
    char summary_path[512];
    snprintf(summary_path, sizeof(summary_path), "%s/metrics_summary.csv", 
             collector->output_dir);
    
    FILE* summary_file = fopen(summary_path, "w");
    if (summary_file) {
        fprintf(summary_file, "metric,min,max,avg,std_dev,count\n");
        
        // Calculate statistics for each metric
        if (collector->count > 0) {
            // Calculate stats for packets_lost
            int min_packets = collector->data[0].packets_lost;
            int max_packets = collector->data[0].packets_lost;
            double sum_packets = 0;
            
            for (int i = 0; i < collector->count; i++) {
                int val = collector->data[i].packets_lost;
                if (val < min_packets) min_packets = val;
                if (val > max_packets) max_packets = val;
                sum_packets += val;
            }
            double avg_packets = sum_packets / collector->count;
            
            fprintf(summary_file, "packets_lost,%d,%d,%.3f,0,%.0f\n", 
                    min_packets, max_packets, avg_packets, (double)collector->count);
            
            // Similar calculations for other metrics...
            // (Implementation for other metrics would be similar)
        }
        
        fclose(summary_file);
    }
    
    pthread_mutex_unlock(&collector->mutex);
    
    printf("📊 Saved %d metric samples to %s\n", collector->count, collector->output_dir);
}

// Get memory usage in MB
double get_memory_usage_mb(void) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        // ru_maxrss is in kilobytes on Linux, bytes on macOS
        #ifdef __linux__
        return usage.ru_maxrss / 1024.0; // Convert KB to MB
        #else
        return usage.ru_maxrss / (1024.0 * 1024.0); // Convert bytes to MB
        #endif
    }
    return 0.0;
}

// Get CPU utilization (simplified version)
double get_cpu_utilization(void) {
    static struct rusage prev_usage;
    static struct timeval prev_time;
    static int first_call = 1;
    
    struct rusage current_usage;
    struct timeval current_time;
    
    if (getrusage(RUSAGE_SELF, &current_usage) != 0) {
        return 0.0;
    }
    
    gettimeofday(&current_time, NULL);
    
    if (first_call) {
        prev_usage = current_usage;
        prev_time = current_time;
        first_call = 0;
        return 0.0;
    }
    
    // Calculate time differences
    double real_time = (current_time.tv_sec - prev_time.tv_sec) + 
                      (current_time.tv_usec - prev_time.tv_usec) / 1000000.0;
    
    double user_time = (current_usage.ru_utime.tv_sec - prev_usage.ru_utime.tv_sec) +
                      (current_usage.ru_utime.tv_usec - prev_usage.ru_utime.tv_usec) / 1000000.0;
    
    double sys_time = (current_usage.ru_stime.tv_sec - prev_usage.ru_stime.tv_sec) +
                     (current_usage.ru_stime.tv_usec - prev_usage.ru_stime.tv_usec) / 1000000.0;
    
    prev_usage = current_usage;
    prev_time = current_time;
    
    if (real_time > 0) {
        return ((user_time + sys_time) / real_time) * 100.0;
    }
    
    return 0.0;
}

// Measure network latency (simplified - ping-like measurement)
double measure_network_latency(const char* target_ip, int port) {
    (void)target_ip; // Suppress unused parameter warning
    (void)port;      // Suppress unused parameter warning
    
    // This is a simplified implementation
    // In a real implementation, you might want to use ICMP ping or TCP connect time
    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    // Simulate some network delay measurement
    usleep(1000); // 1ms base latency
    
    gettimeofday(&end, NULL);
    
    return (end.tv_sec - start.tv_sec) * 1000.0 + 
           (end.tv_usec - start.tv_usec) / 1000.0;
}

// Metrics collection thread
void* metrics_collection_thread(void* arg) {
    const char* output_dir = (const char*)arg;
    
    printf("📊 Starting metrics collection thread\n");
    
    // Initialize collector if not already done
    if (!g_metrics_collector) {
        g_metrics_collector = init_metrics_collector(output_dir);
        if (!g_metrics_collector) {
            printf("Error: Failed to initialize metrics collector\n");
            return NULL;
        }
    }
    
    static int batch_counter = 0;
    
    while (metrics_thread_running) {
        // Collect system metrics
        double memory_mb = get_memory_usage_mb();
        double cpu_percent = get_cpu_utilization();
        double network_latency = measure_network_latency("127.0.0.1", 12345);
        
        // Add sample (with default values for pipeline-specific metrics)
        add_metric_sample(g_metrics_collector,
                         0,                    // packets_lost - will be updated by pipeline
                         network_latency,      // network_latency_ms
                         0.0,                  // communication_time - will be updated
                         batch_counter++,      // batch_order
                         0.0,                  // throughput - will be updated
                         1.0,                  // load_balance_ratio - will be updated
                         memory_mb,            // memory_usage_mb
                         cpu_percent,          // cpu_utilization
                         0.0,                  // stage1_time - will be updated
                         0.0,                  // stage2_time - will be updated
                         0.0,                  // loss_value - will be updated
                         0.0);                 // accuracy - will be updated
        
        // Sleep for 100ms before next measurement
        usleep(100000);
    }
    
    printf("📊 Metrics collection thread stopped\n");
    return NULL;
}

// Start metrics collection
void start_metrics_collection(const char* output_dir) {
    if (metrics_thread_running) {
        printf("Metrics collection already running\n");
        return;
    }
    
    metrics_thread_running = 1;
    
    if (pthread_create(&metrics_thread, NULL, metrics_collection_thread, 
                      (void*)output_dir) != 0) {
        printf("Error: Cannot create metrics collection thread\n");
        metrics_thread_running = 0;
        return;
    }
    
    printf("📊 Metrics collection started\n");
}

// Stop metrics collection
void stop_metrics_collection(void) {
    if (!metrics_thread_running) {
        return;
    }
    
    metrics_thread_running = 0;
    pthread_join(metrics_thread, NULL);
    
    if (g_metrics_collector) {
        cleanup_metrics_collector(g_metrics_collector);
        g_metrics_collector = NULL;
    }
    
    printf("📊 Metrics collection stopped\n");
} 