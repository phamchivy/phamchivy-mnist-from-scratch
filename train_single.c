#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

#include "util/img.h"
#include "neural/nn.h"

double time_in_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    srand(time(NULL));

    printf("🎯 Single Container Training\n\n");

    int number_imgs = 10000;
    printf("Loading %d images...\n", number_imgs);
    Img** imgs = csv_to_imgs("./data/mnist_test.csv", number_imgs);

    printf("Creating neural network (784 -> 300 -> 10)...\n");
    NeuralNetwork* net = network_create(784, 300, 10, 0.1);

    double start = time_in_seconds();

    printf("Starting single training (10 epochs)...\n\n");
    network_train_batch_imgs(net, imgs, number_imgs, 10);

    double end = time_in_seconds();
    printf("\n✅ Training completed in %.2f seconds\n\n", end - start);

    // Test accuracy
    printf("Testing accuracy...\n");
    double accuracy = network_predict_imgs(net, imgs, 1000);
    printf("🎯 Accuracy: %.2f%%\n\n", accuracy * 100);

    // Save model
    printf("Saving model to 'testing_net'...\n");
    network_save(net, "testing_net");

    // Cleanup
    imgs_free(imgs, number_imgs);
    network_free(net);

    printf("✅ Single training completed!\n");
    return 0;
} 