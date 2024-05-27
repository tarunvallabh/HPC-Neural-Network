#ifndef CUDAOPS_H_
#define CUDAOPS_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>


//C(n,m) = A(n,m) + B(n,m)
void add(float *C, float *A, float *B, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            C[i * m + j] = A[i * m + j] + B[i * m + j];
        }
    }
}

//C(n,m) = A(n,m) - B(n,m)
void subtract(float *C, float *A, float *B, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            C[i * m + j] = A[i * m + j] - B[i * m + j];
        }
    }
}


// __global__ void multiplyKernel(float *C, const float *A, const float *B, int n, int p, int m) {
//     int index = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index

//     if (index < n * m) { 
//         int row = index / m;
//         int col = index % m;
//         float temp = 0.0;
//         for (int k = 0; k < p; ++k) {
//             temp += A[row * p + k] * B[k * m + col];
//         }
//         C[index] = temp;
//     }
// }


// void multiply(float *C, const float *A, const float *B, int n, int p, int m) {
//     float *d_A, *d_B, *d_C;
//     size_t sizeA = n * p * sizeof(float);
//     size_t sizeB = p * m * sizeof(float);
//     size_t sizeC = n * m * sizeof(float);

//     // Allocate device memory
//     cudaMalloc((void **)&d_A, sizeA);
//     cudaMalloc((void **)&d_B, sizeB);
//     cudaMalloc((void **)&d_C, sizeC);

//     // Copy matrices from the host to the device
//     cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

//     // Use 128 threads per block
//     int threadsPerBlock = 128;
//     // Calculate the number of blocks needed for the output matrix
//     int totalElements = n * m; // Total number of elements in the output matrix
//     int numBlocks = (totalElements + threadsPerBlock - 1) / threadsPerBlock;


//     // Launch the kernel with a one-dimensional configuration
//     multiplyKernel<<<numBlocks, threadsPerBlock>>>(d_C, d_A, d_B, n, p, m);

//     // Copy the result back to the host
//     cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);

//     // Free device memory
//     cudaFree(d_A);
//     cudaFree(d_B);
//     cudaFree(d_C);
// }


void multiply(float *C, const float *A, const float *B, int n, int p, int m) {
    float *d_A, *d_B, *d_C;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t handle;

    // Allocate device memory
    cudaMalloc((void **)&d_A, n * p * sizeof(float));
    cudaMalloc((void **)&d_B, p * m * sizeof(float));
    cudaMalloc((void **)&d_C, n * m * sizeof(float));

    // Copy matrices from the host to the device
    cudaMemcpy(d_A, A, n * p * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, p * m * sizeof(float), cudaMemcpyHostToDevice);

    // Create a cuBLAS context
    cublasCreate(&handle);

    // Perform matrix multiplication: C = A * B
    // Since CUDA assumes column-major storage and our matrices are row-major,
    // we treat A as B^T and B as A^T, and perform B^T * A^T = (A * B)^T operation.
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, p, &alpha, d_B, m, d_A, p, &beta, d_C, m);

    // Copy the result back to the host
    cudaMemcpy(C, d_C, n * m * sizeof(float), cudaMemcpyDeviceToHost);

    cublasDestroy(handle);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


void multiply_transpose(float *C, const float *A, const float *B, int n, int p, int m) {
    float *d_A, *d_B, *d_C;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t handle;

    // Allocate device memory
    cudaMalloc((void **)&d_A, n * p * sizeof(float));
    cudaMalloc((void **)&d_B, p * m * sizeof(float));
    cudaMalloc((void **)&d_C, n * m * sizeof(float));

    // Copy matrices from the host to the device
    cudaMemcpy(d_A, A, n * p * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, p * m * sizeof(float), cudaMemcpyHostToDevice);

    // Create a cuBLAS context
    cublasCreate(&handle);

    // Perform matrix multiplication: C = A * B
    // Since CUDA assumes column-major storage and our matrices are row-major,
    // we treat A as B^T and B as A^T, and perform B^T * A^T = (A * B)^T operation.
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, p, &alpha, d_B, p, d_A, p, &beta, d_C, m);



    // Copy the result back to the host
    cudaMemcpy(C, d_C, n * m * sizeof(float), cudaMemcpyDeviceToHost);

    cublasDestroy(handle);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


// __global__ void multiplyTransposeKernel(float *C, const float *A, const float *B, int n, int p, int m) {
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     if (index < n * m) {
//         int row = index / m;
//         int col = index % m;
//         float sum = 0.0;
//         for (int k = 0; k < p; ++k) {
//             // Adjusting for B being accessed as transposed
//             sum += A[row * p + k] * B[col * p + k]; // accessed B's column like its a row
//         }
//         C[index] = sum;
//     }
// }

// void multiply_transpose(float *C, const float *A, const float *B, int n, int p, int m) {
//     float *d_A, *d_B, *d_C;
//     size_t sizeA = n * p * sizeof(float);
//     size_t sizeB = m * p * sizeof(float); // Adjusting size for B, considering it as m x p
//     size_t sizeC = n * m * sizeof(float);

//     cudaMalloc((void **)&d_A, sizeA);
//     cudaMalloc((void **)&d_B, sizeB);
//     cudaMalloc((void **)&d_C, sizeC);

//     cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

//     int threadsPerBlock = 128;
//     int totalElements = n * m;
//     int numBlocks = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

//     multiplyTransposeKernel<<<numBlocks, threadsPerBlock>>>(d_C, d_A, d_B, n, p, m);

//     cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);

//     cudaFree(d_A);
//     cudaFree(d_B);
//     cudaFree(d_C);
// }

void transpose_multiply(float *C, const float *A, const float *B, int n, int p, int m) {
    float *d_A, *d_B, *d_C;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t handle;

    // Allocate device memory
    cudaMalloc((void **)&d_A, n * p * sizeof(float));
    cudaMalloc((void **)&d_B, p * m * sizeof(float));
    cudaMalloc((void **)&d_C, n * m * sizeof(float));

    // Copy matrices from the host to the device
    cudaMemcpy(d_A, A, n * p * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, p * m * sizeof(float), cudaMemcpyHostToDevice);

    // Create a cuBLAS context
    cublasCreate(&handle);

    // Perform matrix multiplication: C = A * B
    // Since CUDA assumes column-major storage and our matrices are row-major,
    // we treat A as B^T and B as A^T, and perform B^T * A^T = (A * B)^T operation.
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, p, &alpha, d_B, m, d_A, n, &beta, d_C, m);



    // Copy the result back to the host
    cudaMemcpy(C, d_C, n * m * sizeof(float), cudaMemcpyDeviceToHost);

    cublasDestroy(handle);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// __global__ void transposeMultiplyKernel(float *C, const float *A, const float *B, int n, int p, int m) {
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     if (index < n * m) {
//         int row = index / m; // Corresponds to 'row' in C, and 'col' in A since A is transposed
//         int col = index % m; // Column in B and C
//         float sum = 0.0;
//         for (int k = 0; k < p; ++k) {
//             sum += A[k * n + row] * B[k * m + col]; // k*n+row accesses A's column as if it's a row
//         }
//         C[index] = sum;
//     }
// }

// void transpose_multiply(float *C, const float *A, const float *B, int n, int p, int m) {
//     float *d_A, *d_B, *d_C;
//     size_t sizeA = n * p * sizeof(float); // Size for A
//     size_t sizeB = p * m * sizeof(float); // Size for B
//     size_t sizeC = n * m * sizeof(float); // Size for the result matrix C

//     cudaMalloc((void **)&d_A, sizeA);
//     cudaMalloc((void **)&d_B, sizeB);
//     cudaMalloc((void **)&d_C, sizeC);

//     cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

//     int threadsPerBlock = 128;
//     int totalElements = n * m;
//     int numBlocks = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

//     transposeMultiplyKernel<<<numBlocks, threadsPerBlock>>>(d_C, d_A, d_B, n, p, m);

//     cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);

//     cudaFree(d_A);
//     cudaFree(d_B);
//     cudaFree(d_C);
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


void relu(float *out, float *m1, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            // Apply the ReLU function: f(x) = max(0, x)
            out[i * m + j] = (m1[i * m + j] > 0) ? m1[i * m + j] : 0.0;
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