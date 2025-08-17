#include "nn.h"
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include "../matrix/ops.h"
#include "../neural/activations.h"
#include "../socket/socket_utils.h"

#define MAXCHAR 1000

// 784, 300, 10
// UNCHANGED: network_create modification
NeuralNetwork* network_create(int input, int hidden, int output, double lr) {
    NeuralNetwork* net = malloc(sizeof(NeuralNetwork));
    net->input = input;
    net->hidden = hidden;
    net->output = output;
    net->learning_rate = lr;
    
    // NEW: Initialize EASGD parameters with defaults
    net->alpha = 0.0;           // Will be set explicitly
    net->beta = 0.0;            // Will be set explicitly
    net->easgd_enabled = false; // Disabled by default
    
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
	Matrix* hidden_outputs = apply(sigmoid, hidden_inputs);
	Matrix* final_inputs = dot(net->output_weights, hidden_outputs);
	Matrix* final_outputs = apply(sigmoid, final_inputs);

	// Find errors
	Matrix* output_errors = subtract(output, final_outputs);
	double loss = 0.0;
	for (int i = 0; i < output->rows; i++) {
		double diff = output->entries[i][0] - final_outputs->entries[i][0];
		loss += diff * diff;
	}

	Matrix* transposed_mat = transpose(net->output_weights);
	Matrix* hidden_errors = dot(transposed_mat, output_errors);
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
	Matrix* added_mat = add(net->output_weights, scaled_mat);

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
	net->hidden_weights = added_mat; 

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

void network_train_batch_imgs(NeuralNetwork* net, Img** imgs, int batch_size, int epochs) {
	double loss_sum = 0.0;
    int loss_count = 0;
    for (int epoch = 0; epoch < epochs; epoch++) {
        //double total_loss = 0.0;
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

double time_in_socket_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Keep ALL existing functions unchanged, ADD/MODIFY these:

// NEW: Initialize EASGD parameters
void network_easgd_init(NeuralNetwork* net, double alpha, double beta) {
    net->alpha = alpha;
    net->beta = beta;
    net->easgd_enabled = true;
    printf("EASGD initialized: α=%.3f, β=%.3f\n", alpha, beta);
}

// NEW: Static elastic center (only on master)
static ElasticCenter elastic_center = {NULL, NULL, false, 0};

// NEW: Initialize elastic center with current network weights
void elastic_center_init(NeuralNetwork* net) {
    if (!elastic_center.initialized) {
        elastic_center.center_hidden_weights = matrix_copy(net->hidden_weights);
        elastic_center.center_output_weights = matrix_copy(net->output_weights);
        elastic_center.initialized = true;
        elastic_center.update_count = 0;
        printf("[Master] Elastic center initialized\n");
    }
}

// NEW: Simple elastic center update với 1 worker - w̄ ← w̄ + β(wᵢ - w̄)
void elastic_center_update(NeuralNetwork* net, double* worker_weights, int weight_count) {
    if (!net->easgd_enabled) return;
    
    if (!elastic_center.initialized) {
        elastic_center_init(net);
        return;
    }
    
    // Convert worker weights array to matrices
    Matrix* worker_hidden = matrix_create(net->hidden_weights->rows, net->hidden_weights->cols);
    Matrix* worker_output = matrix_create(net->output_weights->rows, net->output_weights->cols);
    
    int idx = 0;
    // Fill worker hidden weights
    for (int i = 0; i < worker_hidden->rows; i++) {
        for (int j = 0; j < worker_hidden->cols; j++) {
            worker_hidden->entries[i][j] = worker_weights[idx++];
        }
    }
    // Fill worker output weights
    for (int i = 0; i < worker_output->rows; i++) {
        for (int j = 0; j < worker_output->cols; j++) {
            worker_output->entries[i][j] = worker_weights[idx++];
        }
    }
    
    // Update elastic center: w̄ ← w̄ + β(wᵢ - w̄)
    
    // Hidden layer update
    Matrix* hidden_diff = subtract(worker_hidden, elastic_center.center_hidden_weights);
    Matrix* hidden_update = scale(net->beta, hidden_diff);
    Matrix* new_center_hidden = add(elastic_center.center_hidden_weights, hidden_update);
    
    matrix_free(elastic_center.center_hidden_weights);
    elastic_center.center_hidden_weights = new_center_hidden;
    
    // Output layer update
    Matrix* output_diff = subtract(worker_output, elastic_center.center_output_weights);
    Matrix* output_update = scale(net->beta, output_diff);
    Matrix* new_center_output = add(elastic_center.center_output_weights, output_update);
    
    matrix_free(elastic_center.center_output_weights);
    elastic_center.center_output_weights = new_center_output;
    
    elastic_center.update_count++;
    
    // Cleanup
    matrix_free(worker_hidden);
    matrix_free(worker_output);
    matrix_free(hidden_diff);
    matrix_free(hidden_update);
    matrix_free(output_diff);
    matrix_free(output_update);
}

// NEW: Sequential elastic center update với cả master & slaver
void elastic_center_update_sequential(NeuralNetwork* net, double* master_weights, double* slaver_weights, int weight_count) {
    if (!net->easgd_enabled) return;
    
    printf("[Master] Updating elastic center sequentially...\n");
    
    // Update với master weights: w̄ ← w̄ + β(w_master - w̄)
    elastic_center_update(net, master_weights, weight_count);
    printf("[Master] Center updated with master weights\n");
    
    // Update với slaver weights: w̄ ← w̄ + β(w_slaver - w̄)  
    elastic_center_update(net, slaver_weights, weight_count);
    printf("[Master] Center updated with slaver weights\n");
    
    printf("[Master] Sequential elastic center update completed (round %d)\n", 
           elastic_center.update_count);
}

// UNCHANGED: elastic_center_get_weights function
double* elastic_center_get_weights(int* count_out) {
    if (!elastic_center.initialized) {
        *count_out = 0;
        return NULL;
    }
    
    int count = 0;
    count += elastic_center.center_hidden_weights->rows * elastic_center.center_hidden_weights->cols;
    count += elastic_center.center_output_weights->rows * elastic_center.center_output_weights->cols;
    
    double* center_weights = (double*)malloc(sizeof(double) * count);
    int idx = 0;
    
    // Pack hidden weights
    for (int i = 0; i < elastic_center.center_hidden_weights->rows; i++) {
        for (int j = 0; j < elastic_center.center_hidden_weights->cols; j++) {
            center_weights[idx++] = elastic_center.center_hidden_weights->entries[i][j];
        }
    }
    
    // Pack output weights
    for (int i = 0; i < elastic_center.center_output_weights->rows; i++) {
        for (int j = 0; j < elastic_center.center_output_weights->cols; j++) {
            center_weights[idx++] = elastic_center.center_output_weights->entries[i][j];
        }
    }
    
    *count_out = count;
    return center_weights;
}

// UNCHANGED: network_apply_elastic_averaging function
void network_apply_elastic_averaging(NeuralNetwork* net, double* center_weights, int weight_count) {
    if (!net->easgd_enabled) {
        network_set_weights(net, center_weights, weight_count);
        return;
    }
    
    // Convert center weights array to matrices
    Matrix* center_hidden = matrix_create(net->hidden_weights->rows, net->hidden_weights->cols);
    Matrix* center_output = matrix_create(net->output_weights->rows, net->output_weights->cols);
    
    int idx = 0;
    // Fill center hidden weights
    for (int i = 0; i < center_hidden->rows; i++) {
        for (int j = 0; j < center_hidden->cols; j++) {
            center_hidden->entries[i][j] = center_weights[idx++];
        }
    }
    // Fill center output weights
    for (int i = 0; i < center_output->rows; i++) {
        for (int j = 0; j < center_output->cols; j++) {
            center_output->entries[i][j] = center_weights[idx++];
        }
    }
    
    // Apply elastic averaging: wᵢ ← wᵢ + α(w̄ - wᵢ)
    
    // Hidden weights update
    Matrix* hidden_diff = subtract(center_hidden, net->hidden_weights);
    Matrix* hidden_update = scale(net->alpha, hidden_diff);
    Matrix* new_hidden = add(net->hidden_weights, hidden_update);
    
    matrix_free(net->hidden_weights);
    net->hidden_weights = new_hidden;
    
    // Output weights update
    Matrix* output_diff = subtract(center_output, net->output_weights);
    Matrix* output_update = scale(net->alpha, output_diff);
    Matrix* new_output = add(net->output_weights, output_update);
    
    matrix_free(net->output_weights);
    net->output_weights = new_output;
    
    // Cleanup
    matrix_free(center_hidden);
    matrix_free(center_output);
    matrix_free(hidden_diff);
    matrix_free(hidden_update);
    matrix_free(output_diff);
    matrix_free(output_update);
    
    printf("[Worker] Applied elastic averaging with α=%.3f\n", net->alpha);
}

// UNCHANGED: elastic_center_cleanup function
void elastic_center_cleanup(void) {
    if (elastic_center.initialized) {
        matrix_free(elastic_center.center_hidden_weights);
        matrix_free(elastic_center.center_output_weights);
        elastic_center.initialized = false;
        printf("[Master] Elastic center cleaned up\n");
    }
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

// THÊM VÀO CUỐI FILE neural/nn.c

// Static separate elastic centers for hybrid parallelism
static SeparateElasticCenter separate_center = {
    .center_hidden_weights = NULL,
    .center_output_weights = NULL,
    .hidden_initialized = false,
    .output_initialized = false,
    .hidden_update_count = 0,
    .output_update_count = 0
};

void separate_elastic_center_init_hidden(int hidden_rows, int hidden_cols) {
    if (!separate_center.hidden_initialized) {
        separate_center.center_hidden_weights = matrix_create(hidden_rows, hidden_cols);
        matrix_randomize(separate_center.center_hidden_weights, hidden_cols);
        separate_center.hidden_initialized = true;
        separate_center.hidden_update_count = 0;
        printf("[Parameter Server] Hidden elastic center initialized (%dx%d)\n", 
               hidden_rows, hidden_cols);
    }
}

void separate_elastic_center_init_output(int output_rows, int output_cols) {
    if (!separate_center.output_initialized) {
        separate_center.center_output_weights = matrix_create(output_rows, output_cols);
        matrix_randomize(separate_center.center_output_weights, output_cols);
        separate_center.output_initialized = true;
        separate_center.output_update_count = 0;
        printf("[Parameter Server] Output elastic center initialized (%dx%d)\n", 
               output_rows, output_cols);
    }
}

void separate_elastic_center_update_hidden(double* worker_weights, int weight_count, double beta) {
    if (!separate_center.hidden_initialized) {
        printf("[Parameter Server] Hidden center not initialized!\n");
        return;
    }
    
    // Convert worker weights array to matrix
    Matrix* worker_matrix = matrix_create(separate_center.center_hidden_weights->rows, 
                                        separate_center.center_hidden_weights->cols);
    
    int idx = 0;
    for (int i = 0; i < worker_matrix->rows; i++) {
        for (int j = 0; j < worker_matrix->cols; j++) {
            worker_matrix->entries[i][j] = worker_weights[idx++];
        }
    }
    
    // Update elastic center: w̄ ← w̄ + β(w_worker - w̄)
    Matrix* diff = subtract(worker_matrix, separate_center.center_hidden_weights);
    Matrix* update = scale(beta, diff);
    Matrix* new_center = add(separate_center.center_hidden_weights, update);
    
    matrix_free(separate_center.center_hidden_weights);
    separate_center.center_hidden_weights = new_center;
    separate_center.hidden_update_count++;
    
    // Cleanup
    matrix_free(worker_matrix);
    matrix_free(diff);
    matrix_free(update);
}

void separate_elastic_center_update_output(double* worker_weights, int weight_count, double beta) {
    if (!separate_center.output_initialized) {
        printf("[Parameter Server] Output center not initialized!\n");
        return;
    }
    
    // Convert worker weights array to matrix
    Matrix* worker_matrix = matrix_create(separate_center.center_output_weights->rows, 
                                        separate_center.center_output_weights->cols);
    
    int idx = 0;
    for (int i = 0; i < worker_matrix->rows; i++) {
        for (int j = 0; j < worker_matrix->cols; j++) {
            worker_matrix->entries[i][j] = worker_weights[idx++];
        }
    }
    
    // Update elastic center: w̄ ← w̄ + β(w_worker - w̄)
    Matrix* diff = subtract(worker_matrix, separate_center.center_output_weights);
    Matrix* update = scale(beta, diff);
    Matrix* new_center = add(separate_center.center_output_weights, update);
    
    matrix_free(separate_center.center_output_weights);
    separate_center.center_output_weights = new_center;
    separate_center.output_update_count++;
    
    // Cleanup
    matrix_free(worker_matrix);
    matrix_free(diff);
    matrix_free(update);
}

double* separate_elastic_center_get_hidden_weights(int* count_out) {
    if (!separate_center.hidden_initialized) {
        *count_out = 0;
        return NULL;
    }
    
    int count = separate_center.center_hidden_weights->rows * separate_center.center_hidden_weights->cols;
    double* weights = malloc(sizeof(double) * count);
    
    int idx = 0;
    for (int i = 0; i < separate_center.center_hidden_weights->rows; i++) {
        for (int j = 0; j < separate_center.center_hidden_weights->cols; j++) {
            weights[idx++] = separate_center.center_hidden_weights->entries[i][j];
        }
    }
    
    *count_out = count;
    return weights;
}

double* separate_elastic_center_get_output_weights(int* count_out) {
    if (!separate_center.output_initialized) {
        *count_out = 0;
        return NULL;
    }
    
    int count = separate_center.center_output_weights->rows * separate_center.center_output_weights->cols;
    double* weights = malloc(sizeof(double) * count);
    
    int idx = 0;
    for (int i = 0; i < separate_center.center_output_weights->rows; i++) {
        for (int j = 0; j < separate_center.center_output_weights->cols; j++) {
            weights[idx++] = separate_center.center_output_weights->entries[i][j];
        }
    }
    
    *count_out = count;
    return weights;
}

void separate_elastic_center_cleanup(void) {
    if (separate_center.hidden_initialized) {
        matrix_free(separate_center.center_hidden_weights);
        separate_center.hidden_initialized = false;
        printf("[Parameter Server] Hidden elastic center cleaned up\n");
    }
    
    if (separate_center.output_initialized) {
        matrix_free(separate_center.center_output_weights);
        separate_center.output_initialized = false;
        printf("[Parameter Server] Output elastic center cleaned up\n");
    }
}