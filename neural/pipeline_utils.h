#ifndef PIPELINE_UTILS_H
#define PIPELINE_UTILS_H

#include "../matrix/matrix.h"
#include "../neural/nn.h"
#include <stdbool.h>

// Forward declaration from nn.h to avoid circular dependency

// Pipeline utility functions
double sigmoid_activation(double x);
Matrix* apply_sigmoid(Matrix* m);
Matrix* sigmoid_derivative(Matrix* m);
Matrix* matrix_multiply_elementwise(Matrix* a, Matrix* b);
Matrix* matrix_transpose_multiply(Matrix* a, Matrix* b);

// Activation functions for pipeline stages
Matrix* relu_activation(Matrix* input);
Matrix* relu_derivative(Matrix* input);
Matrix* softmax_activation(Matrix* input);

#endif