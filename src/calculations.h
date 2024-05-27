#ifndef CALCULATIONS_H_
#define CALCULATIONS_H_

#include <stdlib.h>
#include <stdio.h>
#include "neuralnet.h"
#include "operations.h"
// #include "cudaops.h"

// JUST UPDATE THE LAYERS' BATCH SIZE FOR THE LAST LAYER WHEN RUNNING THE TRAINING LOOP IN THE MAIN FUNCTION
// or just have a field called batchsize, and use that for all calcs instead of n->layers batchsize

void forwardSingle(nn *n, int l, float *inputArray) {
    multiply(n->layers[l]->z, n->layers[l]->weights, inputArray, n->layers[l]->out, n->layers[l]->in, n->layers[l]->batchSize);

    // add(n->layers[l]->z, n->layers[l]->biases, n->layers[l]->out, n->layers[l]->batchSize);
    add(n->layers[l]->z, n->layers[l]->z, n->layers[l]->biases, n->layers[l]->out, n->layers[l]->batchSize);
    // sigmoid(n->layers[l]->a, n->layers[l]->z, n->layers[l]->out, n->layers[l]->batchSize);
    relu(n->layers[l]->a, n->layers[l]->z, n->layers[l]->out, n->layers[l]->batchSize);
}


void forwardPass(nn *n, float *inputArray) {
    // Forward through first layer
    forwardSingle(n, 1, inputArray);
    // Forward through remaining layers
    for (int l = 2; l < n->num_layers-1; l++) {
        forwardSingle(n, l, n->layers[l-1]->a);
    }

    int last_l = n->num_layers-1;
    multiply(n->layers[last_l]->z, n->layers[last_l]->weights, n->layers[last_l-1]->a, n->layers[last_l]->out, n->layers[last_l]->in, n->layers[last_l]->batchSize);

    // add(n->layers[last_l]->z, n->layers[last_l]->biases, n->layers[last_l]->out, n->layers[last_l]->batchSize);
    add(n->layers[last_l]->z, n->layers[last_l]->z, n->layers[last_l]->biases, n->layers[last_l]->out, n->layers[last_l]->batchSize);
    
    
    softmax(n->layers[last_l]->a, n->layers[last_l]->z, n->layers[last_l]->out, n->layers[last_l]->batchSize);
}


float cross_entropy_loss(nn *n, float *labels, int batch_size) {
    int last_layer_idx = n->num_layers - 1;
    layer *last_layer = n->layers[last_layer_idx];
    
    float cost = 0.0; // Initialize the cost
    int num_classes = last_layer->out; // Assuming the output layer has one unit per class

    // Compute cross-entropy loss
    for (int i = 0; i < batch_size; i++) {
        for (int c = 0; c < num_classes; c++) {
            int idx = i * num_classes + c; // index in the batch for the current sample and class
            float p_c = last_layer->a[idx]; // predicted probability for class c
            float y_c = labels[idx]; // true label for class c
            if (y_c > 0) { // Only add to cost if y_c is 1 (true class)
                cost += -y_c * log(p_c + 1e-8); // Add a small value to avoid log(0)
            }
        }
    }

    cost /= batch_size; // Average the cost over the batch size

    return cost; // Return the cost
}



void compute_cost(nn *n, float *labels, int batch_size) {
    int last_layer_idx = n->num_layers - 1;
    layer *last_layer = n->layers[last_layer_idx];
    
    
    subtract(last_layer->a, last_layer->a, labels, last_layer->out, last_layer->batchSize);
}

void backSingle(float *outputArray, float *inputArray, nn *n, int l, int batch_size) {
    if (l != n->num_layers - 1) {
        // For ReLU layers: Calculate the derivative and multiply by outputArray
        relu_prime(n->layers[l]->z, n->layers[l]->z, n->layers[l]->out, n->layers[l]->batchSize);
        hadamard_product(n->layers[l]->grad_b, outputArray, n->layers[l]->z, n->layers[l]->out, n->layers[l]->batchSize);
    } else {
        // multiply_scalar(n->layers[l]->grad_b, outputArray, 1, n->layers[l]->out, n->layers[l]->batchSize);
        memcpy(n->layers[l]->grad_b, outputArray, n->layers[l]->out * n->layers[l]->batchSize * sizeof(float));

    }
    multiply_transpose(n->layers[l]->grad_w, n->layers[l]->grad_b, inputArray, n->layers[l]->out, n->layers[l]->batchSize, n->layers[l]->in);
}


void backPropogate(nn *n, float *inputArray, int batch_size) {
    int nl = n->num_layers - 1;
    // Back through last layer if no hidden layers
    if (nl == 1) {
        backSingle(n->layers[nl]->a, inputArray, n, nl, batch_size);
    }
    else {
        // Back through last layer
        backSingle(n->layers[nl]->a, n->layers[nl-1]->a, n, nl, batch_size);
        // Back through remaining layers
        for (int l = nl - 1; l > 1; l--) {
            transpose_multiply(n->layers[l]->a, n->layers[l+1]->weights, n->layers[l+1]->grad_b, n->layers[l]->out, n->layers[l+1]->out, n->layers[l]->batchSize);
            backSingle(n->layers[l]->a, n->layers[l-1]->a, n, l, batch_size);
        }
        transpose_multiply(n->layers[1]->a, n->layers[2]->weights, n->layers[2]->grad_b, n->layers[1]->out, n->layers[2]->out, n->layers[1]->batchSize);
        backSingle(n->layers[1]->a, inputArray, n, 1, batch_size);
    }
} 


void train_batch(float *inputArray, float *labels, int batch_size, nn *n) {
    // Print the current batch size
    

    forwardPass(n, inputArray);

    compute_cost(n, labels, batch_size);

    backPropogate(n, inputArray, batch_size);

    free(inputArray);
    // return cost;
}

void update_network(nn *n, float alpha, int batch_size) {
    float scalar = alpha / (float)batch_size;

    for(int l = 1; l < n->num_layers; l++) {
        // update weights and bias
        // multiply_scalar(n->layers[l]->grad_b, scalar, n->layers[l]->out, n->layers[l]->batchSize);

        // native CPU
        multiply_scalar(n->layers[l]->grad_b, n->layers[l]->grad_b, scalar, n->layers[l]->out, n->layers[l]->batchSize);
        multiply_scalar(n->layers[l]->grad_w, n->layers[l]->grad_w, scalar, n->layers[l]->out, n->layers[l]->in);

        // int total_elements = n->layers[l]->out * n->layers[l]->in;
        // cblas_saxpy(n->layers[l]->out * n->layers[l]->in, -scalar, n->layers[l]->grad_w, 1, n->layers[l]->weights, 1);
        // multiply_scalar(n->layers[l]->grad_w, scalar, n->layers[l]->out, n->layers[l]->in);

        subtract(n->layers[l]->weights, n->layers[l]->weights, n->layers[l]->grad_w, n->layers[l]->out, n->layers[l]->in);
        subtract_biases(n->layers[l]->biases, n->layers[l]->grad_b, n->layers[l]->out, batch_size, n->layers[l]->batchSize);
    }
}



#endif