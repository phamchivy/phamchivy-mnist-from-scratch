#include <stdio.h>
#include <stdlib.h>
#include "util/img.h"
#include "neural/nn.h"

int main() {
    int number_test_imgs = 3000;
    Img** test_imgs = csv_to_imgs("./data/mnist_test.csv", number_test_imgs);
    NeuralNetwork* net = network_load("testing_net");

    double score = network_predict_imgs(net, test_imgs, 1000);
    printf("Score: %1.5f\n", score);

    imgs_free(test_imgs, number_test_imgs);
    network_free(net);
    return 0;
}
