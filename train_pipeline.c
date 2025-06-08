#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "pipeline/pipeline_nn.h"
#include "util/img.h"

void print_usage(const char* prog_name) {
    printf("Pipeline Parallelism Usage:\n");
    printf("  %s stage1 <forward_port> <backward_port>\n", prog_name);
    printf("  %s stage2 <forward_port> <backward_port> <stage1_ip>\n", prog_name);
    printf("  %s coordinator <stage1_port> <stage2_port>\n", prog_name);
    printf("\nExample:\n");
    printf("  %s stage1 12345 12346\n", prog_name);
    printf("  %s stage2 12345 12346 172.31.0.2\n", prog_name);
    printf("  %s coordinator 12345 12346\n", prog_name);
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
    double start = time_in_seconds();

    if (strcmp(role, "stage1") == 0) {
        if (argc < 4) {
            print_usage(argv[0]);
            return 1;
        }
        
        int forward_port = atoi(argv[2]);
        int backward_port = atoi(argv[3]);
        
        printf("[MAIN] Starting Stage 1 with forward_port=%d, backward_port=%d\n", 
               forward_port, backward_port);
        fflush(stdout);
        
        pipeline_stage1_main(forward_port, backward_port);
        
    } else if (strcmp(role, "stage2") == 0) {
        if (argc < 5) {
            print_usage(argv[0]);
            return 1;
        }
        
        int forward_port = atoi(argv[2]);
        int backward_port = atoi(argv[3]);
        const char* stage1_ip = argv[4];
        
        printf("[MAIN] Starting Stage 2 with forward_port=%d, backward_port=%d, stage1_ip=%s\n", 
               forward_port, backward_port, stage1_ip);
        fflush(stdout);
        
        pipeline_stage2_main(forward_port, backward_port, stage1_ip);
        
    } else if (strcmp(role, "coordinator") == 0) {
        if (argc < 4) {
            print_usage(argv[0]);
            return 1;
        }
        
        int stage1_port = atoi(argv[2]);
        int stage2_port = atoi(argv[3]);
        
        printf("[MAIN] Starting Coordinator monitoring stage1_port=%d, stage2_port=%d\n", 
               stage1_port, stage2_port);
        fflush(stdout);
        
        coordinator_main(stage1_port, stage2_port);
        
    } else {
        printf("Unknown role: %s\n", role);
        print_usage(argv[0]);
        return 1;
    }

    double end = time_in_seconds();
    printf("[MAIN] %s completed in %.2f seconds.\n", role, end - start);

    return 0;
} 