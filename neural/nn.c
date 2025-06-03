#include "nn.h"
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>
#include "../matrix/ops.h"
#include "../neural/activations.h"
#include "../socket/socket_utils.h"

#define MAXCHAR 1000



// 784, 300, 10
NeuralNetwork* network_create(int input, int hidden, int output, double lr) {
	NeuralNetwork* net = malloc(sizeof(NeuralNetwork));
	net->input = input;
	net->hidden = hidden;
	net->output = output;
	net->learning_rate = lr;
	Matrix* hidden_layer = matrix_create(hidden, input);
	Matrix* output_layer = matrix_create(output, hidden);
	matrix_randomize(hidden_layer, hidden);
	matrix_randomize(output_layer, output);
	net->hidden_weights = hidden_layer;
	net->output_weights = output_layer;
	return net;
}

double network_train(NeuralNetwork* net, Matrix* input, Matrix* output) {
	// Feed forward
	Matrix* hidden_inputs	= dot(net->hidden_weights, input);
	Matrix* hidden_outputs = apply(sigmoid, hidden_inputs); // Tính đầu ra của lớp 1 
	Matrix* final_inputs = dot(net->output_weights, hidden_outputs);
	Matrix* final_outputs = apply(sigmoid, final_inputs); // Tính đầu ra của lớp 2

	// Find errors
	Matrix* output_errors = subtract(output, final_outputs); // Lấy sai số 
	double loss = 0.0;
	for (int i = 0; i < output->rows; i++) {
		double diff = output->entries[i][0] - final_outputs->entries[i][0];
		loss += diff * diff;
	}

	Matrix* transposed_mat = transpose(net->output_weights);
	Matrix* hidden_errors = dot(transposed_mat, output_errors); // Lấy hidden error của output 
	matrix_free(transposed_mat);

	// Backpropogate
	// output_weights = add(
	// 		 output_weights, 
	//     scale(
	// 			  net->lr, 
	// 			  dot(
	// 		 			multiply(
	// 						output_errors, 
	// 				  	sigmoidPrime(final_outputs)
	// 					), 
	// 					transpose(hidden_outputs)
	// 				)
	// 		 )
	// )
	Matrix* sigmoid_primed_mat = sigmoidPrime(final_outputs);
	Matrix* multiplied_mat = multiply(output_errors, sigmoid_primed_mat);
	transposed_mat = transpose(hidden_outputs);
	Matrix* dot_mat = dot(multiplied_mat, transposed_mat);
	Matrix* scaled_mat = scale(net->learning_rate, dot_mat);
	Matrix* added_mat = add(net->output_weights, scaled_mat); // điều chỉnh output weights 

	matrix_free(net->output_weights); // Free the old weights before replacing
	net->output_weights = added_mat;

	matrix_free(sigmoid_primed_mat);
	matrix_free(multiplied_mat);
	matrix_free(transposed_mat);
	matrix_free(dot_mat);
	matrix_free(scaled_mat);

	// hidden_weights = add(
	// 	 net->hidden_weights,
	// 	 scale (
	//			net->learning_rate
	//    	dot (
	//				multiply(
	//					hidden_errors,
	//					sigmoidPrime(hidden_outputs)	
	//				)
	//				transpose(inputs)
	//      )
	// 	 )
	// )
	// Reusing variables after freeing memory
	sigmoid_primed_mat = sigmoidPrime(hidden_outputs);
	multiplied_mat = multiply(hidden_errors, sigmoid_primed_mat);
	transposed_mat = transpose(input);
	dot_mat = dot(multiplied_mat, transposed_mat);
	scaled_mat = scale(net->learning_rate, dot_mat);
	added_mat = add(net->hidden_weights, scaled_mat);
	matrix_free(net->hidden_weights); // Free the old hidden_weights before replacement
	net->hidden_weights = added_mat; // điều chỉnh hidden weights 

	matrix_free(sigmoid_primed_mat);
	matrix_free(multiplied_mat);
	matrix_free(transposed_mat);
	matrix_free(dot_mat);
	matrix_free(scaled_mat);

	// Free matrices
	matrix_free(hidden_inputs);
	matrix_free(hidden_outputs);
	matrix_free(final_inputs);
	matrix_free(final_outputs);
	matrix_free(output_errors);
	matrix_free(hidden_errors);
	return loss;
}

double* network_get_weights(NeuralNetwork* net, int* count_out) {
    int count = 0;

    // Tính tổng số trọng số
    count += net->hidden_weights->rows * net->hidden_weights->cols;
    count += net->output_weights->rows * net->output_weights->cols;

    double* all_weights = (double*)malloc(sizeof(double) * count);
    int idx = 0;

    // Lưu trọng số hidden
    for (int i = 0; i < net->hidden_weights->rows; i++) {
        for (int j = 0; j < net->hidden_weights->cols; j++) {
            all_weights[idx++] = net->hidden_weights->entries[i][j];
        }
    }

    // Lưu trọng số output
    for (int i = 0; i < net->output_weights->rows; i++) {
        for (int j = 0; j < net->output_weights->cols; j++) {
            all_weights[idx++] = net->output_weights->entries[i][j];
        }
    }

    *count_out = count;
    return all_weights;
}

void network_set_weights(NeuralNetwork* net, const double* weights, int count) {
    int idx = 0;

    // Gán lại hidden_weights
    for (int i = 0; i < net->hidden_weights->rows; i++) {
        for (int j = 0; j < net->hidden_weights->cols; j++) {
            net->hidden_weights->entries[i][j] = weights[idx++];
        }
    }

    // Gán lại output_weights
    for (int i = 0; i < net->output_weights->rows; i++) {
        for (int j = 0; j < net->output_weights->cols; j++) {
            net->output_weights->entries[i][j] = weights[idx++];
        }
    }

    if (idx != count) {
        fprintf(stderr, "Warning: mismatch in set_weights (%d vs %d)\n", idx, count);
    }
}

double time_diff(struct timeval start, struct timeval end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
}

double network_train_model_parallelism(NeuralNetwork* net, Matrix* input, Matrix* output, bool is_master) {
    int mid_hidden = net->hidden_weights->rows / 2;
    int mid_output = net->output_weights->rows / 2;

    Matrix* hidden_weights_master = slice_matrix_rows(net->hidden_weights, 0, mid_hidden);
    Matrix* output_weights_master = slice_matrix_rows(net->output_weights, 0, mid_output);
    Matrix* hidden_weights_slave  = slice_matrix_rows(net->hidden_weights, mid_hidden, net->hidden_weights->rows);
    Matrix* output_weights_slave  = slice_matrix_rows(net->output_weights, mid_output, net->output_weights->rows);

    int sockfd;
    if (is_master) {
        sockfd = setup_server(port);
        printf("[Master] Server started. Waiting for connection...\n");
        sockfd = accept_client(sockfd);
        printf("[Master] Connection accepted.\n");
    } else {
        sockfd = connect_to_server(ip, port);
        printf("[Slave] Connected to master.\n");
    }

    if (is_master) {
        // === FORWARD MASTER ===
        Matrix* hidden_inputs_master = dot(hidden_weights_master, input);
        Matrix* hidden_outputs_master = apply(sigmoid, hidden_inputs_master);
        Matrix* final_inputs_master = dot(output_weights_master, hidden_outputs_master);
        Matrix* final_outputs_master = apply(sigmoid, final_inputs_master);

        // === RECEIVE FROM SLAVE ===
        Matrix* hidden_outputs_slave = recv_matrix(sockfd);
        Matrix* final_outputs_slave  = recv_matrix(sockfd);
        double loss_slave = recv_loss(sockfd);

        if (!hidden_outputs_slave || !final_outputs_slave) {
            fprintf(stderr, "[Master] Failed to receive data from slave.\n");
            exit(EXIT_FAILURE);
        }

        // === MERGE & COMPUTE ERROR ===
        Matrix* hidden_outputs = concat_rows(hidden_outputs_master, hidden_outputs_slave);
        Matrix* final_outputs  = concat_rows(final_outputs_master, final_outputs_slave);
        Matrix* output_errors  = subtract(output, final_outputs);

        // === CALCULATE LOSS ===
        double total_loss = 0.0;
        for (int i = 0; i < output->rows; i++) {
            double diff = output->entries[i][0] - final_outputs->entries[i][0];
            total_loss += diff * diff;
        }

        // === BACKPROP ===

        // Output weights update
        Matrix* sigmoid_primed = sigmoidPrime(final_outputs);
        Matrix* delta_output = multiply(output_errors, sigmoid_primed);
        Matrix* trans_hidden = transpose(hidden_outputs);
        Matrix* grad_output = dot(delta_output, trans_hidden);
        Matrix* delta_output_scaled = scale(net->learning_rate, grad_output);
        Matrix* new_output_weights = add(net->output_weights, delta_output_scaled);

        matrix_free(net->output_weights);
        net->output_weights = new_output_weights;

        // Hidden weights update
        Matrix* trans_output_weights = transpose(net->output_weights);
        Matrix* hidden_errors = dot(trans_output_weights, output_errors);
        Matrix* sigmoid_prime_hidden = sigmoidPrime(hidden_outputs);
        Matrix* delta_hidden = multiply(hidden_errors, sigmoid_prime_hidden);
        Matrix* trans_input = transpose(input);
        Matrix* grad_hidden = dot(delta_hidden, trans_input);
        Matrix* delta_hidden_scaled = scale(net->learning_rate, grad_hidden);
        Matrix* new_hidden_weights = add(net->hidden_weights, delta_hidden_scaled);

        matrix_free(net->hidden_weights);
        net->hidden_weights = new_hidden_weights;

        // === FREE ALL ===
        matrix_free(hidden_weights_master);
        matrix_free(output_weights_master);
        matrix_free(hidden_weights_slave);
        matrix_free(output_weights_slave);

        matrix_free(hidden_inputs_master);
        matrix_free(hidden_outputs_master);
        matrix_free(final_inputs_master);
        matrix_free(final_outputs_master);
        matrix_free(hidden_outputs_slave);
        matrix_free(final_outputs_slave);
        matrix_free(hidden_outputs);
        matrix_free(final_outputs);
        matrix_free(output_errors);
        matrix_free(sigmoid_primed);
        matrix_free(delta_output);
        matrix_free(trans_hidden);
        matrix_free(grad_output);
        matrix_free(delta_output_scaled);
        matrix_free(trans_output_weights);
        matrix_free(hidden_errors);
        matrix_free(sigmoid_prime_hidden);
        matrix_free(delta_hidden);
        matrix_free(trans_input);
        matrix_free(grad_hidden);
        matrix_free(delta_hidden_scaled);

        close(sockfd);
        return total_loss;

    } else {
        // === SLAVE FORWARD ===
        Matrix* hidden_inputs = dot(hidden_weights_slave, input);
        Matrix* hidden_outputs = apply(sigmoid, hidden_inputs);
        Matrix* final_inputs = dot(output_weights_slave, hidden_outputs);
        Matrix* final_outputs = apply(sigmoid, final_inputs);

        // === CALCULATE LOSS ===
        double loss = 0.0;
        for (int i = output->rows / 2; i < output->rows; i++) {
            double diff = output->entries[i][0] - final_outputs->entries[i - output->rows / 2][0];
            loss += diff * diff;
        }

        // === SEND TO MASTER ===
        send_matrix(sockfd, hidden_outputs);
        send_matrix(sockfd, final_outputs);
        send_loss(sockfd, loss);

        // === FREE ===
        matrix_free(hidden_weights_master);
        matrix_free(output_weights_master);
        matrix_free(hidden_weights_slave);
        matrix_free(output_weights_slave);

        matrix_free(hidden_inputs);
        matrix_free(hidden_outputs);
        matrix_free(final_inputs);
        matrix_free(final_outputs);

        close(sockfd);
        return loss;
    }
}

void network_train_batch_imgs_model_parallelism(NeuralNetwork* net, Img** imgs, int batch_size, int epochs, bool is_master) {
    double loss_sum = 0.0;
    int loss_count = 0;

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < batch_size; i++) {
            if (is_master && i % 100 == 0) {
                printf("Img No. %d\n", i);
            }

            Img* cur_img = imgs[i];
            Matrix* img_data = matrix_flatten(cur_img->img_data, 0); // Flatten to column
            Matrix* output = matrix_create(10, 1);
            output->entries[cur_img->label][0] = 1;

            // ⬇️ Sử dụng hàm song song mới
            double loss = network_train_model_parallelism(net, img_data, output, is_master);

            if (is_master) {
                loss_sum += loss;
                loss_count++;

                if (i % 1000 == 0 && loss_count > 0) {
                    printf("Average loss after %d images: %.6f\n", i+1, loss_sum / loss_count);
                    loss_sum = 0;
                    loss_count = 0;
                }
            }

            matrix_free(output);
            matrix_free(img_data);
        }

        if (is_master) {
            printf("Epoch %d/%d finished.\n", epoch + 1, epochs);
        }
    }
}

void network_train_batch_imgs(NeuralNetwork* net, Img** imgs, int batch_size, int epochs) {
	double loss_sum = 0.0;
    int loss_count = 0;
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < batch_size; i++) {
            if (i % 100 == 0) printf("Img No. %d\n", i);

            Img* cur_img = imgs[i];
            Matrix* img_data = matrix_flatten(cur_img->img_data, 0); // 0 = flatten to column vector
            Matrix* output = matrix_create(10, 1);
            output->entries[cur_img->label][0] = 1; // Setting the result

            // Train on this image
			double loss = network_train(net, img_data, output); // New version returns loss
			loss_sum += loss;
			loss_count++;

		if (i % 1000 == 0 && loss_count > 0) {
            printf("Average loss after %d images: %.6f\n", i+1, loss_sum / loss_count);
            loss_sum = 0;
            loss_count = 0;
        }

            // Clean up
            matrix_free(output);
            matrix_free(img_data);
        }

        // In ra loss mỗi epoch
        printf("Epoch %d/%d \n", epoch + 1, epochs);
    }
}

void network_train_batch_imgs_socket(
    NeuralNetwork* net,
    Img** imgs,
    int batch_size,
    int epochs,
    bool is_master,
    const char* ip,
    int port,
    int local_cpu_count, // <--- THÊM THAM SỐ NÀY
	Img** test_imgs
) {
    int weight_count;
    double* weights_buffer = NULL;
    int start_index = 0, end_index = batch_size;

	double loss_sum = 0.0;
    int loss_count = 0;

    // Thiết lập socket
    int sockfd;
    int slaver_cpu_count = 0;

    if (is_master) {
        sockfd = setup_server(port);
        printf("[Master] Server started. Waiting for connection...\n");
        sockfd = accept_client(sockfd);
        printf("[Master] Connection accepted.\n");

        // Nhận số core từ slaver
        recv_all(sockfd, &slaver_cpu_count, sizeof(int));
        printf("[Master] Slaver has %d cores\n", slaver_cpu_count);

        int total_cpu = local_cpu_count + slaver_cpu_count;

        // Phân chia dữ liệu theo tỷ lệ số core
        int master_imgs = (batch_size * local_cpu_count) / total_cpu;
        start_index = 0;
        end_index = master_imgs;

    } else {
        sockfd = connect_to_server(ip, port);
        printf("[Slaver] Connected to master.\n");

        // Gửi số core của slaver cho master
        send_all(sockfd, &local_cpu_count, sizeof(int));

        int master_imgs; // sẽ được tính lại cùng công thức master dùng
        recv_all(sockfd, &master_imgs, sizeof(int));  // Nhận từ master

        start_index = master_imgs;
        end_index = batch_size;
    }

    // Master gửi lại cho slaver số ảnh nó xử lý để slaver biết phần còn lại
    if (is_master) {
        send_all(sockfd, &end_index, sizeof(int)); // end_index chính là số ảnh master xử lý
    }

    printf("[%s] Processing images from %d to %d\n", is_master ? "Master" : "Slaver", start_index, end_index);
	fflush(stdout);

    weights_buffer = network_get_weights(net, &weight_count); // để biết weight_count

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = start_index; i < end_index; i++) {
            // Train
            Img* cur_img = imgs[i];
            Matrix* input = matrix_flatten(cur_img->img_data, 0);
            Matrix* output = matrix_create(10, 1);
            output->entries[cur_img->label][0] = 1;

			double loss = network_train(net, input, output); // New version returns loss
			loss_sum += loss;
			loss_count++;

			if (i % 1000 == 0 && loss_count > 0) {
				printf("*** Average loss after %d images: %.6f ***\n", i+1, loss_sum / loss_count);
				fflush(stdout);
				loss_sum = 0;
				loss_count = 0;
			}

			if (i % 1000 == 0) {
				double acc = network_predict_imgs(net, test_imgs, 1000);
				printf("*** After %d images: %.2f%% accuracy ***\n", i, acc * 100);
				fflush(stdout);
			}

            matrix_free(input);
            matrix_free(output);

            // Mỗi 100 ảnh (batch) thì trao đổi trọng số
            if ((((i - start_index + 1) % 10000) == 0) || (i == (end_index - 1))) {
                if (weights_buffer) free(weights_buffer);
                weights_buffer = network_get_weights(net, &weight_count);

                if (is_master) {
                    // Nhận trọng số từ slaver
                    double* slave_weights = (double*)malloc(sizeof(double) * weight_count);
					struct timeval t_start, t_end;
					gettimeofday(&t_start, NULL);
                    recv_all(sockfd, slave_weights, sizeof(double) * weight_count);
                    printf("[Master] Received weights from slaver at img %d\n", i);
					fflush(stdout);
					// printf("[Master] Sample received weights from slaver: ");
					// fflush(stdout);
					// for (int j = 0; j < 5; j++) {
					// 	printf("%.5f ", slave_weights[j]);
					// 	fflush(stdout);
					// }
					// printf("\n");
					// fflush(stdout);

					// printf("[Master] Master weights: ");
					// fflush(stdout);
					// for (int j = 0; j < 5; j++) {
					// 	printf("%.5f ", weights_buffer[j]);
					// 	fflush(stdout);
					// }
					// printf("\n");
					// fflush(stdout);

					// double diff = 0;
					// for (int j = 0; j < weight_count; j++) {
					// 	double d = fabs(slave_weights[j] - weights_buffer[j]);
					// 	diff += d;
					// }
					// printf("[Master] Total weight diff from slaver: %.6f\n", diff);
					// fflush(stdout);

					// printf("[Master] Average loss after %d images and before update weights: %.6f\n", i+1, loss_sum / loss_count);
					// fflush(stdout);

					// double acc_before = network_predict_imgs(net, test_imgs, 1000);
					// printf("[Master] After %d images: %.2f%% accuracy before update weights\n", i, acc_before * 100);
					// fflush(stdout);

                    // Trung bình
                    for (int j = 0; j < weight_count; j++) {
                        weights_buffer[j] = (weights_buffer[j] + slave_weights[j]) / 2.0;
                    }
					// printf("[Master] Averaged weights (sample): ");
					// fflush(stdout);
					// for (int j = 0; j < 5; j++) {
					// 	printf("%.5f ", weights_buffer[j]);
					// 	fflush(stdout);
					// }
					// printf("\n");
					// fflush(stdout);
                    free(slave_weights);

                    // Gửi lại trọng số mới
                    send_all(sockfd, weights_buffer, sizeof(double) * weight_count);
					gettimeofday(&t_end, NULL);
                    printf("[Master] Sent averaged weights to slaver\n");
					fflush(stdout);

					printf("[Master] recv and send weights took %.6f seconds at img %d\n", time_diff(t_start, t_end),i);
					fflush(stdout);

                    network_set_weights(net, weights_buffer, weight_count);

					// printf("[Master] Average loss after %d images and after update weights: %.6f\n", i+1, loss_sum / loss_count);
					// fflush(stdout);

					// double acc_after = network_predict_imgs(net, test_imgs, 1000);
					// printf("[Master] After %d images: %.2f%% accuracy after update weights\n", i, acc_after * 100);
					// fflush(stdout);
                } else {
					// printf("[Slaver] Sample weights before sending to master: ");
					// fflush(stdout);
					// for (int j = 0; j < 5; j++) {
					// 	printf("%.5f ", weights_buffer[j]);
					// 	fflush(stdout);
					// }
					// printf("\n");
					// fflush(stdout);
                    // Gửi trọng số cho master
					// double acc_before = network_predict_imgs(net, test_imgs, 1000);
					// printf("[Slaver] After %d images: %.2f%% accuracy before send weights\n", i, acc_before * 100);
					// fflush(stdout);

					struct timeval t_start, t_end;

					gettimeofday(&t_start, NULL);
                    send_all(sockfd, weights_buffer, sizeof(double) * weight_count);
                    printf("[Slaver] Sent weights to master at img %d\n", i);
					fflush(stdout);

                    // Nhận lại trọng số đã trung bình
                    recv_all(sockfd, weights_buffer, sizeof(double) * weight_count);
                    printf("[Slaver] Received updated weights from master\n");
					fflush(stdout);
					// printf("[Slaver] Received averaged weights from master (sample): ");
					// fflush(stdout);
					// for (int j = 0; j < 5; j++) {
					// 	printf("%.5f ", weights_buffer[j]);
					// 	fflush(stdout);
					// }
					// printf("\n");
                    network_set_weights(net, weights_buffer, weight_count);
					gettimeofday(&t_end, NULL);
					// double acc_after = network_predict_imgs(net, test_imgs, 1000);
					// printf("[Slaver] After %d images: %.2f%% accuracy after receive weights\n", i, acc_after * 100);
					// fflush(stdout);
					printf("[Slaver] send and recv averaged weights took %.6f seconds at img %d\n", time_diff(t_start, t_end),i);
					fflush(stdout);
                }
            }
        }

        printf("[%s] Epoch %d/%d done.\n", is_master ? "Master" : "Slaver", epoch + 1, epochs);
        fflush(stdout);
    }

    free(weights_buffer);
    socket_close(sockfd);
}

Matrix* network_predict_img(NeuralNetwork* net, Img* img) {
	Matrix* img_data = matrix_flatten(img->img_data, 0);
	Matrix* res = network_predict(net, img_data);
	matrix_free(img_data);
	return res;
}

double network_predict_imgs(NeuralNetwork* net, Img** imgs, int n) {
	int n_correct = 0;
	for (int i = 0; i < n; i++) {
		Matrix* prediction = network_predict_img(net, imgs[i]);
		if (matrix_argmax(prediction) == imgs[i]->label) {
			n_correct++;
		}
		matrix_free(prediction);
	}
	return 1.0 * n_correct / n;
}

Matrix* network_predict(NeuralNetwork* net, Matrix* input_data) {
	Matrix* hidden_inputs	= dot(net->hidden_weights, input_data);
	Matrix* hidden_outputs = apply(sigmoid, hidden_inputs);
	Matrix* final_inputs = dot(net->output_weights, hidden_outputs);
	Matrix* final_outputs = apply(sigmoid, final_inputs);
	Matrix* result = softmax(final_outputs);

	matrix_free(hidden_inputs);
	matrix_free(hidden_outputs);
	matrix_free(final_inputs);
	matrix_free(final_outputs);

	return result;
}

void network_save(NeuralNetwork* net, char* file_string) {
	mkdir(file_string, 0777);
	// Write the descriptor file
	chdir(file_string);
	FILE* descriptor = fopen("descriptor", "w");
	fprintf(descriptor, "%d\n", net->input);
	fprintf(descriptor, "%d\n", net->hidden);
	fprintf(descriptor, "%d\n", net->output);
	fclose(descriptor);
	matrix_save(net->hidden_weights, "hidden");
	matrix_save(net->output_weights, "output");
	printf("Successfully written to '%s'\n", file_string);
	chdir("-"); // Go back to the orignal directory
}

NeuralNetwork* network_load(char* file_string) {
	NeuralNetwork* net = malloc(sizeof(NeuralNetwork));
	char entry[MAXCHAR];
	chdir(file_string);

	FILE* descriptor = fopen("descriptor", "r");
	fgets(entry, MAXCHAR, descriptor);
	net->input = atoi(entry);
	fgets(entry, MAXCHAR, descriptor);
	net->hidden = atoi(entry);
	fgets(entry, MAXCHAR, descriptor);
	net->output = atoi(entry);
	fclose(descriptor);
	net->hidden_weights = matrix_load("hidden");
	net->output_weights = matrix_load("output");
	printf("Successfully loaded network from '%s'\n", file_string);
	chdir("-"); // Go back to the original directory
	return net;
}

void network_print(NeuralNetwork* net) {
	printf("# of Inputs: %d\n", net->input);
	printf("# of Hidden: %d\n", net->hidden);
	printf("# of Output: %d\n", net->output);
	printf("Hidden Weights: \n");
	matrix_print(net->hidden_weights);
	printf("Output Weights: \n");
	matrix_print(net->output_weights);
}

void network_free(NeuralNetwork *net) {
	matrix_free(net->hidden_weights);
	matrix_free(net->output_weights);
	free(net);
	net = NULL;
}