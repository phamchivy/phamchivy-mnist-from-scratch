#include "ops.h"
#include <stdlib.h>
#include <stdio.h>

// OPTIMIZED: Use 2 threads for 2 cores (1:1 ratio)
#define NUM_THREADS_OPTIMIZED 2

int check_dimensions_opt(Matrix *m1, Matrix *m2) {
	if (m1->rows == m2->rows && m1->cols == m2->cols) return 1;
	return 0;
}

Matrix* multiply_optimized(Matrix *m1, Matrix *m2) {
	if (check_dimensions_opt(m1, m2)) {
		Matrix *m = matrix_create(m1->rows, m1->cols);
#		pragma omp parallel for num_threads(NUM_THREADS_OPTIMIZED) collapse(2)
		for (int i = 0; i < m1->rows; i++) {
			for (int j = 0; j < m2->cols; j++) {
				m->entries[i][j] = m1->entries[i][j] * m2->entries[i][j];
			}
		}
		return m;
	} else {
		printf("Dimension mistmatch multiply: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}

Matrix* add_optimized(Matrix *m1, Matrix *m2) {
	if (check_dimensions_opt(m1, m2)) {
		Matrix *m = matrix_create(m1->rows, m1->cols);
#		pragma omp parallel for num_threads(NUM_THREADS_OPTIMIZED) collapse(2)
		for (int i = 0; i < m1->rows; i++) {
			for (int j = 0; j < m2->cols; j++) {
				m->entries[i][j] = m1->entries[i][j] + m2->entries[i][j];
			}
		}
		return m;
	} else {
		printf("Dimension mistmatch add: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}

Matrix* subtract_optimized(Matrix *m1, Matrix *m2) {
	if (check_dimensions_opt(m1, m2)) {
		Matrix *m = matrix_create(m1->rows, m1->cols);
#		pragma omp parallel for num_threads(NUM_THREADS_OPTIMIZED) collapse(2)
		for (int i = 0; i < m1->rows; i++) {
			for (int j = 0; j < m2->cols; j++) {
				m->entries[i][j] = m1->entries[i][j] - m2->entries[i][j];
			}
		}
		return m;
	} else {
		printf("Dimension mistmatch subtract: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}

Matrix* apply_optimized(double (*func)(double), Matrix* m) {
	Matrix *mat = matrix_copy(m);
#	pragma omp parallel for num_threads(NUM_THREADS_OPTIMIZED) collapse(2)
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			mat->entries[i][j] = (*func)(m->entries[i][j]);
		}
	}
	return mat;
}

Matrix* dot_optimized(Matrix *m1, Matrix *m2) {
	if (m1->cols == m2->rows) {
		Matrix *m = matrix_create(m1->rows, m2->cols);
		
		// IMPORTANT: For matrix multiplication, we need to be careful about memory access patterns
		// This version optimizes for cache locality
#		pragma omp parallel for num_threads(NUM_THREADS_OPTIMIZED)
		for (int i = 0; i < m1->rows; i++) {
			for (int j = 0; j < m2->cols; j++) {
				double sum = 0;
				// Inner loop should be sequential for better cache performance
				for (int k = 0; k < m2->rows; k++) {
					sum += m1->entries[i][k] * m2->entries[k][j];
				}
				m->entries[i][j] = sum;
			}
		}
		return m;
	} else {
		printf("Dimension mistmatch dot: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}

Matrix* scale_optimized(double n, Matrix* m) {
	Matrix* mat = matrix_copy(m);
#	pragma omp parallel for num_threads(NUM_THREADS_OPTIMIZED) collapse(2)
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			mat->entries[i][j] *= n;
		}
	}
	return mat;
}

Matrix* addScalar_optimized(double n, Matrix* m) {
	Matrix* mat = matrix_copy(m);
#	pragma omp parallel for num_threads(NUM_THREADS_OPTIMIZED) collapse(2)
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			mat->entries[i][j] += n;
		}
	}
	return mat;
}

Matrix* transpose_optimized(Matrix* m) {
	Matrix* mat = matrix_create(m->cols, m->rows);
#	pragma omp parallel for num_threads(NUM_THREADS_OPTIMIZED) collapse(2)
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			mat->entries[j][i] = m->entries[i][j];
		}
	}
	return mat;
} 