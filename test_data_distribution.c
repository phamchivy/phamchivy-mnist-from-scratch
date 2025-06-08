#include <stdio.h>
#include <stdlib.h>
#include "util/img.h"

int main() {
    printf("Testing MNIST data distribution...\n\n");
    
    int number_imgs = 1000;
    Img** imgs = csv_to_imgs("./data/mnist_test.csv", number_imgs);
    
    // Count distribution in first half (master's data)
    int master_counts[10] = {0};
    for (int i = 0; i < number_imgs/2; i++) {
        master_counts[imgs[i]->label]++;
    }
    
    // Count distribution in second half (slaver's data)
    int slaver_counts[10] = {0};
    for (int i = number_imgs/2; i < number_imgs; i++) {
        slaver_counts[imgs[i]->label]++;
    }
    
    printf("Master's data distribution (samples 0-%d):\n", number_imgs/2-1);
    for (int i = 0; i < 10; i++) {
        printf("Digit %d: %d samples\n", i, master_counts[i]);
    }
    
    printf("\nSlaver's data distribution (samples %d-%d):\n", number_imgs/2, number_imgs-1);
    for (int i = 0; i < 10; i++) {
        printf("Digit %d: %d samples\n", i, slaver_counts[i]);
    }
    
    // Check if distribution is balanced
    printf("\nBalance check:\n");
    for (int i = 0; i < 10; i++) {
        float ratio = (float)master_counts[i] / (master_counts[i] + slaver_counts[i]);
        printf("Digit %d: Master has %.1f%%, Slaver has %.1f%%\n", 
               i, ratio*100, (1-ratio)*100);
    }
    
    imgs_free(imgs, number_imgs);
    return 0;
} 