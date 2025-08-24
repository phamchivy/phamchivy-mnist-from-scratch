// config/config_loader.h
#ifndef CONFIG_LOADER_H
#define CONFIG_LOADER_H

#include <stdbool.h>

// Configuration structures
typedef struct {
    int epochs;
    int batch_size;
    int total_images;
    int test_images;
    double learning_rate;
} TrainingConfig;

typedef struct {
    double alpha;
    double beta;
    bool enabled;
} EASGDConfig;

typedef struct {
    int input_size;
    int hidden_size;
    int output_size;
    char activation[32];
} NetworkConfig;

typedef struct {
    int num_groups;
    int num_stages;
    int sync_frequency;
} PipelineConfig;

typedef struct {
    char ip[32];
    int port;
    int expected_requests;
    int timeout_seconds;
} ServerConfig;

typedef struct {
    int group_id;
    char ip[32];
    char next_stage_ip[32];
    int next_stage_port;
    int listen_port;
} WorkerConfig;

typedef struct {
    char train_path[256];
    char test_path[256];
    double validation_split;
    bool shuffle;
} DataConfig;

typedef struct {
    char level[16];
    bool log_sync_details;
    bool log_timing;
    bool log_loss;
    bool save_logs;
    char log_dir[256];
} LoggingConfig;

typedef struct {
    int save_frequency;
    bool save_individual_stages;
    bool save_final_hybrid;
    char output_dir[256];
} ModelConfig;

// Main configuration structure
typedef struct {
    TrainingConfig training;
    EASGDConfig easgd;
    NetworkConfig network;
    PipelineConfig pipeline;
    ServerConfig server;
    WorkerConfig stage1_workers[4];
    WorkerConfig stage2_workers[4];
    DataConfig data;
    LoggingConfig logging;
    ModelConfig model;
} HybridConfig;

// Function declarations
HybridConfig* load_config(const char* config_file);
void free_config(HybridConfig* config);
void print_config(const HybridConfig* config);
bool validate_config(const HybridConfig* config);

#endif