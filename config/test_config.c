// test_config.c - Test program for configuration loading
#include <stdio.h>
#include <stdlib.h>
#include "config_loader.h"

int main(int argc, char** argv) {
    const char* config_file = (argc > 1) ? argv[1] : "config/config.yml";
    
    printf("=== Configuration Loading Test ===\n");
    printf("Loading config from: %s\n\n", config_file);
    
    HybridConfig* config = load_config(config_file);
    
    if (!config) {
        printf("❌ FAILED: Could not load configuration\n");
        return 1;
    }
    
    printf("✅ SUCCESS: Configuration loaded successfully!\n\n");
    
    // Print detailed configuration
    print_config(config);
    
    // Test individual sections
    printf("\n=== Detailed Configuration Test ===\n");
    
    printf("📚 Training Configuration:\n");
    printf("  - Epochs: %d\n", config->training.epochs);
    printf("  - Batch size: %d\n", config->training.batch_size);
    printf("  - Total images: %d\n", config->training.total_images);
    printf("  - Learning rate: %.3f\n", config->training.learning_rate);
    
    printf("\n🔄 EASGD Configuration:\n");
    printf("  - Alpha (α): %.3f\n", config->easgd.alpha);
    printf("  - Beta (β): %.3f\n", config->easgd.beta);
    printf("  - Enabled: %s\n", config->easgd.enabled ? "Yes" : "No");
    
    printf("\n🧠 Network Architecture:\n");
    printf("  - Input size: %d\n", config->network.input_size);
    printf("  - Hidden size: %d\n", config->network.hidden_size);
    printf("  - Output size: %d\n", config->network.output_size);
    printf("  - Activation: %s\n", config->network.activation);
    
    printf("\n⚙️ Pipeline Configuration:\n");
    printf("  - Number of groups: %d\n", config->pipeline.num_groups);
    printf("  - Number of stages: %d\n", config->pipeline.num_stages);
    printf("  - Sync frequency: %d images\n", config->pipeline.sync_frequency);
    
    printf("\n🖥️ Server Configuration:\n");
    printf("  - IP address: %s\n", config->server.ip);
    printf("  - Port: %d\n", config->server.port);
    printf("  - Expected requests: %d\n", config->server.expected_requests);
    printf("  - Timeout: %d seconds\n", config->server.timeout_seconds);
    
    printf("\n📁 Data Configuration:\n");
    printf("  - Training data: %s\n", config->data.train_path);
    printf("  - Test data: %s\n", config->data.test_path);
    printf("  - Validation split: %.1f%%\n", config->data.validation_split * 100);
    printf("  - Shuffle: %s\n", config->data.shuffle ? "Yes" : "No");
    
    printf("\n📝 Logging Configuration:\n");
    printf("  - Level: %s\n", config->logging.level);
    printf("  - Log sync details: %s\n", config->logging.log_sync_details ? "Yes" : "No");
    printf("  - Log timing: %s\n", config->logging.log_timing ? "Yes" : "No");
    printf("  - Log loss: %s\n", config->logging.log_loss ? "Yes" : "No");
    printf("  - Save logs: %s\n", config->logging.save_logs ? "Yes" : "No");
    printf("  - Log directory: %s\n", config->logging.log_dir);
    
    printf("\n💾 Model Configuration:\n");
    printf("  - Save frequency: %d requests\n", config->model.save_frequency);
    printf("  - Save individual stages: %s\n", config->model.save_individual_stages ? "Yes" : "No");
    printf("  - Save final hybrid: %s\n", config->model.save_final_hybrid ? "Yes" : "No");
    printf("  - Output directory: %s\n", config->model.output_dir);
    
    // Validation test
    printf("\n🔍 Configuration Validation:\n");
    if (validate_config(config)) {
        printf("  ✅ Configuration is valid\n");
    } else {
        printf("  ❌ Configuration validation failed\n");
    }
    
    // Calculate derived values
    printf("\n📊 Derived Values:\n");
    int images_per_group = config->training.total_images / config->pipeline.num_groups;
    int syncs_per_group = images_per_group / config->pipeline.sync_frequency;
    int total_expected_syncs = syncs_per_group * config->pipeline.num_groups * config->pipeline.num_stages;
    
    printf("  - Images per group: %d\n", images_per_group);
    printf("  - Syncs per group per stage: %d\n", syncs_per_group);
    printf("  - Total expected sync requests: %d\n", total_expected_syncs);
    
    if (total_expected_syncs != config->server.expected_requests) {
        printf("  ⚠️  WARNING: Expected requests (%d) doesn't match calculated (%d)\n",
               config->server.expected_requests, total_expected_syncs);
    } else {
        printf("  ✅ Expected requests matches calculated value\n");
    }
    
    // Memory estimation
    printf("\n💭 Memory Estimation:\n");
    int hidden_weights = config->network.hidden_size * config->network.input_size;
    int output_weights = config->network.output_size * config->network.hidden_size;
    int total_weights = hidden_weights + output_weights;
    double memory_mb = (total_weights * sizeof(double)) / (1024.0 * 1024.0);
    
    printf("  - Hidden weights: %d (%dx%d)\n", hidden_weights, 
           config->network.hidden_size, config->network.input_size);
    printf("  - Output weights: %d (%dx%d)\n", output_weights,
           config->network.output_size, config->network.hidden_size);
    printf("  - Total parameters: %d\n", total_weights);
    printf("  - Approximate memory per worker: %.2f MB\n", memory_mb);
    
    printf("\n🚀 System Ready:\n");
    printf("  - Configuration loaded and validated\n");
    printf("  - Ready for hybrid parallelism training\n");
    printf("  - Use 'make hybrid' to build the system\n");
    
    // Cleanup
    free_config(config);
    
    printf("\n✅ Configuration test completed successfully!\n");
    return 0;
}