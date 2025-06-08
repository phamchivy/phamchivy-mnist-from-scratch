#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

#include "util/img.h"
#include "neural/nn.h"

extern void network_train_batch_imgs_socket(NeuralNetwork* net, Img** imgs, int batch_size, int epochs, bool is_master, const char* ip, int port);

void print_usage(const char* prog_name) {
    printf("Usage:\n");
    printf("  %s master <port>\n", prog_name);
    printf("  %s slaver <port> <master_ip>\n", prog_name);
}

double time_in_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    srand(time(NULL));

    const char* role = argv[1];
    int port = atoi(argv[2]);

    int number_imgs = 10000;
    Img** imgs = csv_to_imgs("./data/mnist_test.csv", number_imgs);

    NeuralNetwork* net = network_create(784, 300, 10, 0.1);

    double start = time_in_seconds();

    if (strcmp(role, "master") == 0) {
        printf("[MASTER] Starting on port %d\n", port);
        fflush(stdout);
        network_train_batch_imgs_socket(net, imgs, number_imgs, 10, true, NULL, port);
    } else if (strcmp(role, "slaver") == 0) {
        if (argc < 4) {
            print_usage(argv[0]);
            return 1;
        }
        const char* master_ip = argv[3];
        printf("[SLAVER] Connecting to %s:%d\n", master_ip, port);
        fflush(stdout);
        network_train_batch_imgs_socket(net, imgs, number_imgs, 10, false, master_ip, port);
    } else {
        print_usage(argv[0]);
        return 1;
    }

    double end = time_in_seconds();
    printf("[%s] Training took %.2f seconds.\n", role, end - start);

    if (strcmp(role, "master") == 0) {
        network_save(net, "testing_net");
    }

    imgs_free(imgs, number_imgs);
    network_free(net);

    return 0;
}
