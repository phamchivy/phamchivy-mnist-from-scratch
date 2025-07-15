#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>

#include "util/img.h"
#include "neural/nn.h"

double time_in_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Custom training function with metrics tracking
void network_train_batch_imgs_with_metrics(NeuralNetwork* net, Img** imgs, int batch_size, int epochs) {
    double total_loss = 0.0;
    int total_correct = 0;
    int total_processed = 0;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        printf("=== Epoch %d/%d ===\n", epoch + 1, epochs);
        double epoch_loss = 0.0;
        int epoch_correct = 0;
        int epoch_processed = 0;
        
        for (int i = 0; i < batch_size; i++) {
            if (i % 1000 == 0) {
                printf("Processing image %d/%d...\n", i + 1, batch_size);
            }

            Img* cur_img = imgs[i];
            Matrix* img_data = matrix_flatten(cur_img->img_data, 0); // 0 = flatten to column vector
            Matrix* target_output = matrix_create(10, 1);
            target_output->entries[cur_img->label][0] = 1; // One-hot encoding

            // Train on this image FIRST
            network_train(net, img_data, target_output);
            
            // THEN get prediction after training (for accuracy calculation)
            Matrix* prediction = network_predict(net, img_data);
            int predicted_label = matrix_argmax(prediction);
            if (predicted_label == cur_img->label) {
                epoch_correct++;
            }

            // Calculate loss (simplified cross-entropy)
            double loss = 0.0;
            for (int j = 0; j < 10; j++) {
                if (j == cur_img->label) {
                    loss -= log(prediction->entries[j][0] + 1e-15); // Add small epsilon to avoid log(0)
                }
            }
            epoch_loss += loss;
            epoch_processed++;

            // Clean up
            matrix_free(target_output);
            matrix_free(img_data);
            matrix_free(prediction);
        }
        
        double epoch_accuracy = (double)epoch_correct / epoch_processed * 100.0;
        printf("Epoch %d completed - Average Loss: %.4f, Accuracy: %.2f%% (%d/%d)\n\n", 
               epoch + 1, epoch_loss / epoch_processed, epoch_accuracy, epoch_correct, epoch_processed);
        
        total_loss += epoch_loss;
        total_correct += epoch_correct;
        total_processed += epoch_processed;
    }
    
    printf("=== Training Summary ===\n");
    printf("📉 Final Average Loss: %.4f\n", total_loss / total_processed);
    printf("📊 Final Training Accuracy: %.2f%% (%d/%d)\n", 
           (double)total_correct / total_processed * 100.0, total_correct, total_processed);
}

int main() {
    srand(time(NULL));

    printf("🎯 Single Container Training (Traditional Neural Network)\n");
    printf("Architecture: 784 -> 512 -> 10 (monolithic, same as pipeline equivalent)\n\n");

    // Use same parameters as pipeline for fair comparison
    int number_imgs = 60000;  // Full MNIST training dataset
    int epochs = 5;           // Same as pipeline
    double learning_rate = 0.01; // Same as pipeline
    
    printf("Loading %d images from MNIST training dataset...\n", number_imgs);
    Img** imgs = csv_to_imgs("./data/mnist_train.csv", number_imgs);
    if (!imgs) {
        printf("❌ Error loading training dataset\n");
        return 1;
    }

    printf("Creating traditional neural network (784 -> 512 -> 10)...\n");
    printf("Learning rate: %.3f, Epochs: %d\n\n", learning_rate, epochs);
    
    // Create traditional monolithic neural network
    // Note: Using 512 hidden units to match pipeline's stage1 output size
    NeuralNetwork* net = network_create(784, 512, 10, learning_rate);
    
    if (!net) {
        printf("❌ Error creating neural network\n");
        imgs_free(imgs, number_imgs);
        return 1;
    }

    double start_time = time_in_seconds();

    printf("Starting traditional training...\n");
    printf("(This processes one image at a time sequentially)\n\n");

    // Train using custom function with metrics
    network_train_batch_imgs_with_metrics(net, imgs, number_imgs, epochs);

    double end_time = time_in_seconds();
    double training_time = end_time - start_time;
    
    printf("\n=== Training Completed ===\n");
    printf("⏱️  Total time: %.2f seconds\n", training_time);
    printf("🚀 Throughput: %.2f samples/second\n", number_imgs * epochs / training_time);

    // Test on a subset of data for comparison
    printf("\n=== Testing on Test Dataset ===\n");
    printf("Loading test data...\n");
    Img** test_imgs = csv_to_imgs("./data/mnist_test.csv", 1000);
    if (test_imgs) {
        double test_accuracy = network_predict_imgs(net, test_imgs, 1000);
        printf("🎯 Test accuracy: %.2f%% (%.0f/1000)\n", test_accuracy * 100, test_accuracy * 1000);
        imgs_free(test_imgs, 1000);
    } else {
        printf("⚠️  Could not load test data for evaluation\n");
    }

    // Save model for comparison
    printf("\nSaving model to 'single_training_model'...\n");
    network_save(net, "single_training_model");

    // Cleanup
    network_free(net);
    imgs_free(imgs, number_imgs);

    printf("\n✅ Single container training completed!\n");
    printf("Compare this time with pipeline training results:\n");
    printf("  - Pipeline should be faster due to parallelism\n");
    printf("  - Single container uses less memory\n");
    printf("  - Single container has better cache locality\n");
    printf("  - Accuracy should be similar\n");
    
    return 0;
} 