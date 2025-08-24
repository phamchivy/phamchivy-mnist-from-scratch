// config/config_loader.c
#include "config_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_LINE 1024
#define MAX_KEY 64
#define MAX_VALUE 256

// Helper function to trim whitespace
char* trim_whitespace(char* str) {
    char* end;
    
    // Trim leading space
    while(isspace((unsigned char)*str)) str++;
    
    if(*str == 0) return str;
    
    // Trim trailing space
    end = str + strlen(str) - 1;
    while(end > str && isspace((unsigned char)*end)) end--;
    
    end[1] = '\0';
    return str;
}

// Parse key-value pair from YAML-like format
int parse_key_value(const char* line, char* key, char* value) {
    const char* colon = strchr(line, ':');
    if (!colon) return 0;
    
    // Extract key
    int key_len = colon - line;
    strncpy(key, line, key_len);
    key[key_len] = '\0';
    strcpy(key, trim_whitespace(key));
    
    // Extract value
    strcpy(value, colon + 1);
    strcpy(value, trim_whitespace(value));
    
    // Remove quotes if present
    if (value[0] == '"' && value[strlen(value)-1] == '"') {
        value[strlen(value)-1] = '\0';
        memmove(value, value + 1, strlen(value));
    }
    
    return 1;
}

// Convert string to boolean
bool str_to_bool(const char* str) {
    return (strcmp(str, "true") == 0 || strcmp(str, "True") == 0 || 
            strcmp(str, "TRUE") == 0 || strcmp(str, "1") == 0);
}

HybridConfig* load_config(const char* config_file) {
    FILE* file = fopen(config_file, "r");
    if (!file) {
        printf("Error: Cannot open config file: %s\n", config_file);
        return NULL;
    }
    
    HybridConfig* config = calloc(1, sizeof(HybridConfig));
    if (!config) {
        fclose(file);
        return NULL;
    }
    
    // Set default values
    config->training.epochs = 1;
    config->training.batch_size = 1000;
    config->training.total_images = 60000;
    config->training.test_images = 10000;
    config->training.learning_rate = 0.1;
    
    config->easgd.alpha = 0.01;
    config->easgd.beta = 0.001;
    config->easgd.enabled = true;
    
    config->network.input_size = 784;
    config->network.hidden_size = 300;
    config->network.output_size = 10;
    strcpy(config->network.activation, "sigmoid");
    
    config->pipeline.num_groups = 2;
    config->pipeline.num_stages = 2;
    config->pipeline.sync_frequency = 1000;
    
    strcpy(config->server.ip, "172.32.0.10");
    config->server.port = 12345;
    config->server.expected_requests = 120;
    config->server.timeout_seconds = 3600;
    
    strcpy(config->data.train_path, "./data/mnist_train.csv");
    strcpy(config->data.test_path, "./data/mnist_test.csv");
    config->data.validation_split = 0.1;
    config->data.shuffle = true;
    
    strcpy(config->logging.level, "INFO");
    config->logging.log_sync_details = true;
    config->logging.log_timing = true;
    config->logging.log_loss = true;
    config->logging.save_logs = true;
    strcpy(config->logging.log_dir, "./logs");
    
    config->model.save_frequency = 1000;
    config->model.save_individual_stages = true;
    config->model.save_final_hybrid = true;
    strcpy(config->model.output_dir, "./results");
    
    char line[MAX_LINE];
    char current_section[MAX_KEY] = "";
    int worker_stage1_idx = 0;
    int worker_stage2_idx = 0;
    
    while (fgets(line, sizeof(line), file)) {
        char* trimmed = trim_whitespace(line);
        
        // Skip empty lines and comments
        if (strlen(trimmed) == 0 || trimmed[0] == '#') continue;
        
        char key[MAX_KEY], value[MAX_VALUE];
        
        // Check if it's a section header
        if (strchr(trimmed, ':') && !strchr(trimmed, ' ') && 
            trimmed[strlen(trimmed)-1] == ':') {
            strncpy(current_section, trimmed, strlen(trimmed)-1);
            current_section[strlen(trimmed)-1] = '\0';
            continue;
        }
        
        if (!parse_key_value(trimmed, key, value)) continue;
        
        // Parse based on current section
        if (strcmp(current_section, "training") == 0) {
            if (strcmp(key, "epochs") == 0) config->training.epochs = atoi(value);
            else if (strcmp(key, "batch_size") == 0) config->training.batch_size = atoi(value);
            else if (strcmp(key, "total_images") == 0) config->training.total_images = atoi(value);
            else if (strcmp(key, "test_images") == 0) config->training.test_images = atoi(value);
            else if (strcmp(key, "learning_rate") == 0) config->training.learning_rate = atof(value);
        }
        else if (strcmp(current_section, "easgd") == 0) {
            if (strcmp(key, "alpha") == 0) config->easgd.alpha = atof(value);
            else if (strcmp(key, "beta") == 0) config->easgd.beta = atof(value);
            else if (strcmp(key, "enabled") == 0) config->easgd.enabled = str_to_bool(value);
        }
        else if (strcmp(current_section, "network") == 0) {
            if (strcmp(key, "input_size") == 0) config->network.input_size = atoi(value);
            else if (strcmp(key, "hidden_size") == 0) config->network.hidden_size = atoi(value);
            else if (strcmp(key, "output_size") == 0) config->network.output_size = atoi(value);
            else if (strcmp(key, "activation") == 0) strcpy(config->network.activation, value);
        }
        else if (strcmp(current_section, "pipeline") == 0) {
            if (strcmp(key, "num_groups") == 0) config->pipeline.num_groups = atoi(value);
            else if (strcmp(key, "num_stages") == 0) config->pipeline.num_stages = atoi(value);
            else if (strcmp(key, "sync_frequency") == 0) config->pipeline.sync_frequency = atoi(value);
        }
        else if (strcmp(current_section, "server") == 0) {
            if (strcmp(key, "ip") == 0) strcpy(config->server.ip, value);
            else if (strcmp(key, "port") == 0) {
                config->server.port = atoi(value);
                printf("DEBUG: Set server port to: %d\n", config->server.port);
            }
            else if (strcmp(key, "expected_requests") == 0) config->server.expected_requests = atoi(value);
            else if (strcmp(key, "timeout_seconds") == 0) config->server.timeout_seconds = atoi(value);
        }
        // Add more sections as needed...
    }
    
    fclose(file);
    
    if (!validate_config(config)) {
        free_config(config);
        return NULL;
    }
    
    return config;
}

void free_config(HybridConfig* config) {
    if (config) {
        free(config);
    }
}

void print_config(const HybridConfig* config) {
    if (!config) return;
    
    printf("=== Hybrid Parallelism Configuration ===\n");
    printf("Training: epochs=%d, batch_size=%d, lr=%.3f\n", 
           config->training.epochs, config->training.batch_size, config->training.learning_rate);
    printf("EASGD: alpha=%.3f, beta=%.3f, enabled=%s\n", 
           config->easgd.alpha, config->easgd.beta, config->easgd.enabled ? "true" : "false");
    printf("Network: %d->%d->%d (%s)\n", 
           config->network.input_size, config->network.hidden_size, 
           config->network.output_size, config->network.activation);
    printf("Pipeline: %d groups, %d stages, sync every %d images\n", 
           config->pipeline.num_groups, config->pipeline.num_stages, config->pipeline.sync_frequency);
    printf("Server: %s:%d, expecting %d requests\n", 
           config->server.ip, config->server.port, config->server.expected_requests);
    printf("=========================================\n");
}

bool validate_config(const HybridConfig* config) {
    if (!config) return false;
    
    // Validate ranges
    if (config->training.epochs <= 0 || config->training.batch_size <= 0 || 
        config->training.learning_rate <= 0) {
        printf("Error: Invalid training parameters\n");
        return false;
    }
    
    if (config->easgd.alpha <= 0 || config->easgd.beta <= 0) {
        printf("Error: Invalid EASGD parameters\n");  
        return false;
    }
    
    if (config->network.input_size <= 0 || config->network.hidden_size <= 0 || 
        config->network.output_size <= 0) {
        printf("Error: Invalid network architecture\n");
        return false;
    }
    
    if (config->server.port <= 0 || config->server.port > 65535) {
        printf("Error: Invalid server port\n");
        return false;
    }
    
    return true;
}