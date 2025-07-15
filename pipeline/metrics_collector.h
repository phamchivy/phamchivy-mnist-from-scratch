#ifndef METRICS_COLLECTOR_H
#define METRICS_COLLECTOR_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/resource.h>
#endif

// Time-series metrics structure
typedef struct {
    double timestamp;               // Thời gian (giây từ lúc bắt đầu)
    int packets_lost;              // Số gói tin bị mất
    double network_latency_ms;     // Độ trễ mạng (ms)
    double communication_time;     // Thời gian truyền thông (ms)
    int batch_order;               // Thứ tự batch processing
    double throughput;             // Throughput (samples/second)
    double load_balance_ratio;     // Cân bằng tải giữa các stage
    double memory_usage_mb;        // Sử dụng memory (MB)
    double cpu_utilization;        // Sử dụng CPU (%)
    double stage1_time;            // Thời gian xử lý stage 1
    double stage2_time;            // Thời gian xử lý stage 2
    double loss_value;             // Giá trị loss hiện tại
    double accuracy;               // Độ chính xác hiện tại
} TimeSeriesMetric;

// Metrics collector structure
typedef struct {
    TimeSeriesMetric* data;        // Mảng dữ liệu time-series
    int capacity;                  // Dung lượng mảng
    int count;                     // Số lượng records hiện tại
    pthread_mutex_t mutex;         // Mutex để thread-safe
    FILE* csv_files[9];           // File handles cho từng metric
    char output_dir[256];         // Thư mục output
    struct timeval start_time;    // Thời gian bắt đầu experiment
} MetricsCollector;

// Global metrics collector
extern MetricsCollector* g_metrics_collector;

// Function declarations
MetricsCollector* init_metrics_collector(const char* output_dir);
void cleanup_metrics_collector(MetricsCollector* collector);

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
                     double accuracy);

void save_timeseries_csv(MetricsCollector* collector);
double get_metrics_timestamp(MetricsCollector* collector);

// System metrics collection
double get_memory_usage_mb(void);
double get_cpu_utilization(void);
double measure_network_latency(const char* target_ip, int port);

// Async metrics collection thread
void* metrics_collection_thread(void* arg);
void start_metrics_collection(const char* output_dir);
void stop_metrics_collection(void);

#endif // METRICS_COLLECTOR_H 