#define _GNU_SOURCE
#include <sched.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

#include "util/img.h"
#include "neural/nn.h"

void log_allowed_cpus() {
    cpu_set_t mask;
    CPU_ZERO(&mask);

    if (sched_getaffinity(0, sizeof(cpu_set_t), &mask) == -1) {
        perror("sched_getaffinity");
        fflush(stdout);
        return;
    }

    printf("Allowed CPU cores: ");
    for (int i = 0; i < CPU_SETSIZE; i++) {
        if (CPU_ISSET(i, &mask)) {
            printf("%d ", i);
            fflush(stdout);
        }
    }
    printf("\n");
    fflush(stdout);
}


int get_allowed_cpu_count() {
    cpu_set_t mask;
    CPU_ZERO(&mask);

    if (sched_getaffinity(0, sizeof(cpu_set_t), &mask) == -1) {
        perror("sched_getaffinity");
        return -1;
    }

    int count = 0;
    for (int i = 0; i < CPU_SETSIZE; i++) {
        if (CPU_ISSET(i, &mask)) {
            count++;
        }
    }

    return count;
}

void network_train_batch_imgs_model_parallelism(NeuralNetwork* net, Img** imgs, int batch_size, int epochs, bool is_master);

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

    int number_imgs = 60000;
    Img** imgs = csv_to_imgs("./data/mnist_train.csv", number_imgs);
    Img** test_imgs = csv_to_imgs("./data/mnist_test.csv", 1000);

    NeuralNetwork* net = network_create(784, 300, 10, 0.1);

    double start = time_in_seconds();

    if (strcmp(role, "master") == 0) {
        printf("[MASTER] Starting on port %d\n", port);
        fflush(stdout);
        log_allowed_cpus();
        int cpu_count = get_allowed_cpu_count();
        printf("[%s] Allowed CPU cores: %d\n", role, cpu_count);
        fflush(stdout);
        network_train_batch_imgs_model_parallelism(net, imgs, number_imgs, 1, true)
    } else if (strcmp(role, "slaver") == 0) {
        if (argc < 4) {
            print_usage(argv[0]);
            return 1;
        }
        const char* master_ip = argv[3];
        log_allowed_cpus();
        int cpu_count = get_allowed_cpu_count();
        printf("[%s] Allowed CPU cores: %d\n", role, cpu_count);
        printf("[SLAVER] Connecting to %s:%d\n", master_ip, port);
        fflush(stdout);
        network_train_batch_imgs_model_parallelism(net, imgs, number_imgs, 1, false)
    } else {
        print_usage(argv[0]);
        return 1;
    }

    double end = time_in_seconds();
    printf("[%s] Training took %.2f seconds.\n", role, end - start);

    //if (strcmp(role, "master") == 0) {
        network_save(net, "testing_net");
    //}

    imgs_free(imgs, number_imgs);
    imgs_free(test_imgs, 1000);
    network_free(net);

    return 0;
}
