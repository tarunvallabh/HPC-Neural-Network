#ifndef OPERATIONS_H_
#define OPERATIONS_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>


void add(float *out, float *m1, float *m2, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            out[i * m + j] = m1[i * m + j] + m2[i * m + j];
        }
    }
}


void subtract(float *out, float *m1, float *m2, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            out[i * m + j] = m1[i * m + j] - m2[i * m + j];
        }
    }
}



void multiply(float *out, float *m1, float *m2, int n, int p, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            float sum = 0;
            for (int k = 0; k < p; k++) {
                sum += m1[i * p + k] * m2[k * m + j];
          
            }
            out[i * m + j] = sum;
        }
    }
}


// void multiply(float *out, float *m1, float *m2, int n, int p, int m) {
//     // Parameters: Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, out, ldout
//     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, m, p, 1.0, m1, p, m2, m, 0.0, out, m);
// }





void transpose_multiply(float *out, float *m1, float *m2, int n, int p, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            float sum = 0;
            for (int k = 0; k < p; k++) {
                sum += m1[k * n + i] * m2[k * m + j];
            }
            out[i * m + j] = sum;
        }
    }
}


// void transpose_multiply(float *out, float *m1, float *m2, int n, int p, int m) {
//     // Here A is transposed, so its leading dimension is now n
//     cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, m, p, 1.0, m1, n, m2, m, 0.0, out, m);
// }



void multiply_transpose(float *out, float *m1, float *m2, int n, int p, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            float sum = 0;
            for (int k = 0; k < p; k++) {
                sum += m1[i * p + k] * m2[j * p + k];
            }
            out[i * m + j] = sum;
        }
    }
}

// void multiply_transpose(float *out, float *m1, float *m2, int n, int p, int m) {
//     // Since B is transposed, its leading dimension changes to m
//     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, p, 1.0, m1, p, m2, p, 0.0, out, m);
// }


void hadamard_product(float *out, float *m2, float *m1, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            out[i * m + j] = m1[i * m + j] * m2[i * m + j];
        }
    }
}


void multiply_scalar(float *out, float *m1, float scalar, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            out[i * m + j] = m1[i * m + j] * scalar;
        }
    }
}


// void multiply_scalar(float *m1, float scalar, int n, int m) {
//     // Calculate the total number of elements in the matrix
//     int total_elements = n * m;

//     // Directly scale m1 in-place.
//     cblas_sscal(total_elements, scalar, m1, 1);
// }





void sigmoid(float *out, float *m1, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            // Directly use exp() for the sigmoid function calculation
            out[i * m + j] = 1.0 / (1.0 + exp(-m1[i * m + j]));
        }
    }
}

void relu(float *out, float *m1, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            // Apply the ReLU function: f(x) = max(0, x)
            out[i * m + j] = (m1[i * m + j] > 0) ? m1[i * m + j] : 0.0;
        }
    }
}

void sigmoid_prime(float *out, float *m1, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            float sig = 1.0 / (1.0 + exp(-m1[i * m + j]));
            out[i * m + j] = sig * (1 - sig); // Direct calculation of sigmoid prime
        }
    }
}

void relu_prime(float *out, float *m1, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            // If A[i * m + j] > 0, the derivative is 1; otherwise, it's 0.
            out[i * m + j] = m1[i * m + j] > 0 ? 1.0 : 0.0;
        }
    }
}

void softmax(float *out, float *m1, int n, int m) {
    // Iterate over each column (m columns, each corresponding to a batch example)
    for (int j = 0; j < m; j++) {
        float sum_exp = 0.0;
        // First, find the sum of the exponentials of the scores for the current column
        for (int i = 0; i < n; i++) {
            sum_exp += exp(m1[i * m + j]);
        }
        // Then, compute the softmax for each score in the column
        for (int i = 0; i < n; i++) {
            out[i * m + j] = exp(m1[i * m + j]) / sum_exp;
        }
    }
}


// Derivative of categorical cross-entropy loss with respect to inputs
void cross_entropy_derivative(float *grad, float *probs, float *true_labels, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            int idx = i * m + j;
            grad[idx] = probs[idx] - true_labels[idx];
        }
    }
}



// add bias to all in the batch
void add_bias(float *out, float *m2, int n, int m) {
    // Iterate over each row
    for (int i = 0; i < n; i++) {
        // Iterate over each column in the row
        for (int j = 0; j < m; j++) {
            // C[i*m + j] accesses the element in the ith row and jth column
            // Add the bias B[j] to the current element
            out[i*m + j] += m2[j];
        }
    }
}



// subtract the sum of biases from each row
void subtract_biases(float *m1, float *m2, int n, int batch_size, int m) {
    // Iterate over each row
    for (int i = 0; i < n; i++) {
        float sum = 0;
        // Sum up the biases for the current row in m2
        for (int j = 0; j < batch_size; j++) {
            sum += m2[i * batch_size + j];
        }
        for (int j = 0; j < m; j++) {
            // Subtract the calculated sum from each element
            m1[i * m + j] -= sum;
        }
    }
}


#endif