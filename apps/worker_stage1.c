#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "../neural/nn.h"
#include "../util/img.h"
#include "../socket/socket_utils.h"
#include "../socket/pipeline_socket.h"
#include "../config/config_loader.h"

int main(int argc, char** argv) {
    // if (argc < 7) {
    //     printf("Usage: %s <group_id> <next_stage_ip> <next_stage_port> <server_ip> <server_port>\n", argv[0]);
    //     return 1;
    // }

    if (argc < 6) {
        printf("Usage: %s <config_file> <group_id> <next_stage_ip> <next_stage_port> <server_ip> <server_port>\n", argv[0]);
        printf("   OR: %s <group_id> <next_stage_ip> <next_stage_port> <server_ip> <server_port> (uses default config.yml)\n", argv[0]);
        return 1;
    }

    // Load configuration
    HybridConfig* config;
    int group_id, next_stage_port, server_port;
    char next_stage_ip[64], server_ip[64];
    
    // int group_id = atoi(argv[2]);
    // const char* next_stage_ip = argv[3];
    // int next_stage_port = atoi(argv[4]);
    // const char* server_ip = argv[5];
    // int server_port = atoi(argv[6]);

    if (argc == 6) {
        // Old format - use default config
        config = load_config("config.yml");
        group_id = atoi(argv[1]);
        strcpy(next_stage_ip, argv[2]);
        next_stage_port = atoi(argv[3]);
        strcpy(server_ip, argv[4]);
        server_port = atoi(argv[5]);
    } else {
        // New format - config file specified
        config = load_config(argv[1]);
        group_id = atoi(argv[2]);
        strcpy(next_stage_ip, argv[3]);
        next_stage_port = atoi(argv[4]);
        strcpy(server_ip, argv[5]);
        server_port = atoi(argv[6]);
    }

    if (!config) {
        printf("Failed to load configuration\n");
        return 1;
    }
    
    // printf("[Stage1 Group %d] Starting, next stage: %s:%d, server: %s:%d\n", 
    //        group_id, next_stage_ip, next_stage_port, server_ip, server_port);
    // fflush(stdout);

    printf("[Stage1 Group %d] Starting with config-driven parameters\n", group_id);
    printf("[Stage1 Group %d] EASGD: α=%.3f, β=%.3f, sync_freq=%d\n", 
           group_id, config->easgd.alpha, config->easgd.beta, config->pipeline.sync_frequency);
    printf("[Stage1 Group %d] Next stage: %s:%d, server: %s:%d\n", 
           group_id, next_stage_ip, next_stage_port, server_ip, server_port);
    fflush(stdout);
    
    srand(time(NULL) + group_id);
    sleep(2);
    
    // Load data
    // printf("[Stage1 Group %d] Loading MNIST data...\n", group_id);
    // int number_imgs = 60000;
    // Img** imgs = csv_to_imgs("./data/mnist_train.csv", number_imgs);

    // Load data using config path
    printf("[Stage1 Group %d] Loading MNIST data from %s...\n", group_id, config->data.train_path);
    Img** imgs = csv_to_imgs(config->data.train_path, config->training.total_images);

    // Data partitioning based on group
    int images_per_group = config->training.total_images / config->pipeline.num_groups;
    int start_index = (group_id - 1) * images_per_group;
    int end_index = group_id * images_per_group;
    
    // Data partitioning based on group
    // int start_index = (group_id - 1) * (number_imgs / 2);
    // int end_index = group_id * (number_imgs / 2);
    
    printf("[Stage1 Group %d] Processing images %d to %d (%d total)\n", 
           group_id, start_index, end_index-1, end_index - start_index);
    fflush(stdout);
    
    // Initialize pipeline stage 1 (784 → 300)
    // PipelineStage* stage1 = pipeline_stage_create(784, 300, 0.1);
    // stage1->easgd_enabled = true;
    // stage1->alpha = 0.01;
    // stage1->beta = 0.001;

    PipelineStage* stage1 = pipeline_stage_create(config->network.input_size, 
                                                 config->network.hidden_size, 
                                                 config->training.learning_rate);
    stage1->easgd_enabled = config->easgd.enabled;
    stage1->alpha = config->easgd.alpha;
    stage1->beta = config->easgd.beta;
    
    // printf("[Stage1 Group %d] Pipeline stage initialized: 784→300, α=0.01, β=0.001\n", group_id);
    // fflush(stdout);

    printf("[Stage1 Group %d] Pipeline stage initialized: %d→%d, lr=%.3f, α=%.3f, β=%.3f\n", 
           group_id, config->network.input_size, config->network.hidden_size,
           config->training.learning_rate, config->easgd.alpha, config->easgd.beta);
    fflush(stdout);
    
    // Training loop
    int sync_count = 0;
    double start_time = time_in_socket_seconds();
    
    for (int epoch = 0; epoch < config->training.epochs; epoch++) {
        // printf("[Stage1 Group %d] Starting epoch %d\n", group_id, epoch + 1);
        // fflush(stdout);

        printf("[Stage1 Group %d] Starting epoch %d/%d\n", group_id, epoch + 1, config->training.epochs);
        fflush(stdout);
        
        for (int i = start_index; i < end_index; i++) {
            Img* cur_img = imgs[i];
            Matrix* input = matrix_flatten(cur_img->img_data, 0);
            
            // Forward pass through stage 1
            Matrix* hidden_outputs = pipeline_stage1_forward(stage1, input);
            
            // Send activation to stage 2
            int stage2_sock = connect_to_next_stage(next_stage_ip, next_stage_port);
            if (stage2_sock < 0) {
                printf("[Stage1 Group %d] Failed to connect to stage 2\n", group_id);
                matrix_free(input);
                matrix_free(hidden_outputs);
                continue;
            }
            
            // Send image label và activation
            send_all(stage2_sock, &cur_img->label, sizeof(int));
            send_activation(stage2_sock, hidden_outputs);
            
            // Receive gradient from stage 2
            //Matrix* grad_from_stage2 = receive_gradient(stage2_sock, 300, 1);
            Matrix* grad_from_stage2 = receive_gradient(stage2_sock, config->network.hidden_size, 1);
            close(stage2_sock);
            
            if (grad_from_stage2 != NULL) {
                // Backward pass
                pipeline_stage1_backward(stage1, input, hidden_outputs, grad_from_stage2);
                matrix_free(grad_from_stage2);
            }
            
            // Cleanup
            matrix_free(input);
            matrix_free(hidden_outputs);
            
            // Sync every 1000 images (independent of stage 2)
            if ((i - start_index + 1) % config->pipeline.sync_frequency == 0) {
                printf("[Stage1 Group %d] Syncing after %d images (sync #%d)\n", 
                       group_id, i - start_index + 1, ++sync_count);
                
                // Get stage 1 weights
                int weight_count;
                double* weights = pipeline_stage1_get_weights(stage1, &weight_count);
                
                // Send to parameter server
                if (send_weights_to_parameter_server(server_ip, server_port, group_id, 
                                                   WEIGHT_TYPE_HIDDEN, weights, weight_count) == 0) {
                    if (config->logging.log_sync_details) {
                        printf("[Stage1 Group %d] Successfully synced weights\n", group_id);
                    }
                } else {
                    printf("[Stage1 Group %d] Failed to sync weights\n", group_id);
                }
                
                free(weights);
            }
            
            if ((i - start_index + 1) % config->pipeline.sync_frequency == 0) {
                printf("[Stage1 Group %d] Processed %d images\n", group_id, i - start_index + 1);
                fflush(stdout);
            }
        }
        
        printf("[Stage1 Group %d] Epoch %d completed\n", group_id, epoch + 1);
        fflush(stdout);
    }
    
    double end_time = time_in_socket_seconds();
    printf("[Stage1 Group %d] Training completed in %.2f seconds, total syncs: %d\n", 
           group_id, end_time - start_time, sync_count);
    fflush(stdout);
    
    // Save final model
    // THAY THẾ đoạn save cuối file worker_stage1.c:
    // Save final model
    pipeline_stage_save(stage1, "stage1", group_id);
    printf("[Stage1 Group %d] Final model saved\n", group_id);

    // if (config->model.save_individual_stages) {
    //     char stage_path[512];
    //     snprintf(stage_path, sizeof(stage_path), "%s/stage1_group_%d", config->model.output_dir, group_id);
        
    //     // Create output directory
    //     char mkdir_cmd[512];
    //     snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p %s", config->model.output_dir);
    //     system(mkdir_cmd);
        
    //     pipeline_stage_save(stage1, "stage1", group_id);
    //     printf("[Stage1 Group %d] Final model saved to %s\n", group_id, stage_path);
    // }
    
    // Cleanup
    imgs_free(imgs, config->training.total_images);
    pipeline_stage_free(stage1);
    
    return 0;
}