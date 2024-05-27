#ifndef NEURALNET_H_
#define NEURALNET_H_


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct layer {
    int in;
    int out;
    int batchSize;
    float *weights; 
    float *biases;
    float *a;
    float *z;
    float *grad_b;
    float *grad_w;
} layer;


typedef struct nn {
    int input_size;
    int output_size;
    int batchSize;
    int num_layers;
    int *layer_sizes;
    layer **layers;
} nn;

layer *createLayer(int in, int out, int batchSize) {
    layer *l = (layer *)malloc(sizeof(layer));
    if(!l) printf("Failed to allocate memory for layer\n");

    l->in = in;
    l->out = out;
    l->batchSize = batchSize;

    l->weights = (float *)malloc(out * in * sizeof(float));
    // make biases a matrix for easier operation later
    l->biases = (float *)malloc(out * batchSize * sizeof(float));


    // activations
    l->a = (float *)malloc(out * batchSize * sizeof(float));
    // z layers
    l->z = (float *)malloc(out * batchSize * sizeof(float));
    // gradients of biases
    l->grad_b = (float *)malloc(out * batchSize * sizeof(float));

    // gradient of weights
    l->grad_w = (float *)malloc(out * in * sizeof(float));

    return l;
}

nn *createNetwork(int in, int out, int batchSize, int num_layers, int *layer_sizes) {
    nn *neuralNetwork = (nn *)malloc(sizeof(nn));
    neuralNetwork->input_size = in;
    neuralNetwork->output_size = out;
    neuralNetwork->batchSize = batchSize;
    neuralNetwork->num_layers = num_layers;
    neuralNetwork->layer_sizes = layer_sizes;
    neuralNetwork->layers = (layer **)malloc(num_layers * sizeof(layer *));

    // intialize first layer to null as input layer
    neuralNetwork->layers[0] = NULL;
    for(int l = 1; l < num_layers; l++) {
        neuralNetwork->layers[l] = createLayer(layer_sizes[l-1], layer_sizes[l], batchSize);
    }
    return neuralNetwork;
}



// Function to initialize a layer with weights using the Kaiming initialization scheme
void initializeLayer(layer *l) {
    float sigma = sqrt(2.0 / (float) l->in); // Standard deviation for Kaiming initialization
    for (int i = 0; i < l->out; ++i) {
        for (int j = 0; j < l->in; ++j) {
            // Use Box-Muller transform to generate two Gaussian random numbers
            float u1 = (float)rand() / RAND_MAX;
            float u2 = (float)rand() / RAND_MAX;
            float radius = sqrt(-2 * log(u1));
            float theta = 2 * M_PI * u2;

            // Generate the random weight with the Kaiming initialization formula
            float z0 = radius * cos(theta) * sigma;

            // Assign the weight to the current position
            l->weights[i * l->in + j] = z0;
        }
    }

    // Initialize biases to zero
    for (int i = 0; i < l->out * l->batchSize; i++) {
        l->biases[i] = 0.0;
    }
}


void initializeNetwork(nn *neuralNetwork) {
    // initalize layers of the network
    for (int l = 1; l < neuralNetwork->num_layers; l++) {
        initializeLayer(neuralNetwork->layers[l]);
    }
}

// need this to reset the grads
void initializeGradients(layer *l) {
    for (int i = 0; i < l->out * l->batchSize; i++) {
        l->grad_b[i] = 0.0;
    }
    for (int i = 0; i < l->out * l->in; i++) {
        l->grad_w[i] = 0.0;
    }
}

// fo this for each layer
void initializeBatch(nn *neuralNetwork) {
    for (int l = 1; l < neuralNetwork->num_layers; l++) {
        initializeGradients(neuralNetwork->layers[l]);
    }
}


void free_layer(layer *l) {
    free(l->grad_b);
    free(l->grad_w);
    free(l->weights);
    free(l->biases);
    free(l->z);
    free(l->a);
    free(l);
}

void free_nn(nn *neuralNetwork) {
    for(int l = 1; l < neuralNetwork->num_layers; l++) {
        free_layer(neuralNetwork->layers[l]);
    }
    free(neuralNetwork->layers);
    free(neuralNetwork);
}



#endif