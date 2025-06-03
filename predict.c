#include <stdio.h>
#include <stdlib.h>
#include "util/img.h"
#include "neural/nn.h"

int main() {
    int number_test_imgs = 10000;
    Img** test_imgs = csv_to_imgs("./data/mnist_test.csv", number_test_imgs);
    NeuralNetwork* net = network_load("testing_net_master");

        // Test: In thử 20 ảnh đầu
    for (int i = 0; i < 20; i++) {
        printf("\n=== Ảnh #%d ===\n", i);
        printf("Nhãn đúng (label): %d\n", test_imgs[i]->label);

        Matrix* prediction = network_predict_img(net, test_imgs[i]);
        int predicted_label = matrix_argmax(prediction);
        printf("Model dự đoán: %d\n", predicted_label);

        if (predicted_label == test_imgs[i]->label) {
            printf("✅ Đúng\n");
        } else {
            printf("❌ Sai\n");
        }

        matrix_free(prediction);

        // Nếu muốn in ra hình ảnh để hình dung (có thể bỏ qua nếu không cần)
    }


    double score = network_predict_imgs(net, test_imgs, 10000);
    printf("Score: %1.5f\n", score);

    imgs_free(test_imgs, number_test_imgs);
    network_free(net);
    return 0;
}
