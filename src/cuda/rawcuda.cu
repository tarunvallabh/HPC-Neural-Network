#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h> 
#include <time.h>
#include "cudaops.h"
#include "neuralnet.h"
#include "cudacalcs.h"

//timing variables 
clock_t startTrainingTime, endTrainingTime;
clock_t startInferenceTime, endInferenceTime;
float grindRate;

int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

//CHANGE THIS TO JUST READ IT AND STORE IT IN COLUMN MAJOR FORMAT, THEN WE CAN JUST BATCH IT WAY EASIER. 

float* readMNISTLabels(const char* filename, int* number_of_labels) {
    FILE* file = fopen(filename, "rb");
    if (file) {
        int magic_number = 0;
        fread(&magic_number, sizeof(magic_number), 1, file);
        magic_number = reverseInt(magic_number);

        int num_labels = 0;
        fread(&num_labels, sizeof(num_labels), 1, file);
        num_labels = reverseInt(num_labels);

        *number_of_labels = num_labels;

        // Allocating a 1D array instead of a float pointer
        // Size is num_labels * 10 because each label is a one-hot encoded vector of size 10
        float* labels = (float*)calloc(num_labels * 10, sizeof(float));

        for (int i = 0; i < num_labels; i++) {
            unsigned char temp = 0;
            fread(&temp, sizeof(temp), 1, file);
            // Calculating the index in the 1D array where the one-hot encoding starts for this label
            // Then setting the appropriate position to 1.0
            labels[i * 10 + temp] = 1.0;
        }
        fclose(file);
        return labels;
    } else {
        printf("Error opening file\n");
        return NULL;
    }
}


float* readMNISTImages(const char* filename, int* number_of_images, int* size) {
    FILE* file = fopen(filename, "rb");
    if (file) {
        int magic_number = 0;
        int num_images = 0;
        int rows = 0;
        int cols = 0;
        fread(&magic_number, sizeof(magic_number), 1, file);
        magic_number = reverseInt(magic_number);
        fread(&num_images, sizeof(num_images), 1, file);
        num_images = reverseInt(num_images);
        fread(&rows, sizeof(rows), 1, file);
        rows = reverseInt(rows);
        fread(&cols, sizeof(cols), 1, file);
        cols = reverseInt(cols);

        *number_of_images = num_images;
        *size = rows * cols;

        // Allocate memory for all images
        float* images = (float*)malloc(num_images * rows * cols * sizeof(float));

        // Read each image and normalize
        for(int i = 0; i < num_images; i++) {
            for(int j = 0; j < rows * cols; j++) {
                unsigned char temp = 0;
                fread(&temp, sizeof(temp), 1, file);
                // Normalize pixel values to the range [0, 1]
                images[i * rows * cols + j] = temp/255.0;
            }
        }
        fclose(file);
        return images;
    } else {
        printf("Error opening file\n");
        return NULL;
    }
}


// Helper function to transpose the images
void transposeImages(float* src, float* dst, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}


float* processBatch(float *images, int numberOfImages, int inputSize, int batchNum, int batchSize, int currentBatchSize) {
    int startImageIndex = batchNum * batchSize; // Calculate the starting index of the batch

    // Allocate memory for the transposed batch images
    float *transposedBatchImages = (float*)calloc(batchSize * inputSize, sizeof(float));

    // Directly write images to the transposed batch array in column-major order
    for (int i = 0; i < currentBatchSize; i++) {
        for (int j = 0; j < inputSize; j++) {
            int imageIndex = (startImageIndex + i) * inputSize + j;
            if (imageIndex < numberOfImages * inputSize) { // Ensure we don't read past the image array
                // Calculate the index in the transposed array
                int transposedIndex = j * batchSize + i; // Use batchSize here for consistent memory layout
                transposedBatchImages[transposedIndex] = images[imageIndex];
            }
        }
    }

    // Note: There might be extra zero-padding for batches that are not full, which should be handled appropriately.

    return transposedBatchImages; // Return the pointer to the transposed images for this batch
}


float* processLabelsBatch(float *labels, int numberOfLabels, int batchNum, int batchSize, int currentBatchSize) {
    int startLabelIndex = batchNum * batchSize;

    // Allocate memory for the transposed batch of labels with potential padding
    // Ensuring that the allocated memory is for batchSize * 10, not currentBatchSize, to include padding
    float *transposedBatchLabels = (float*)calloc(batchSize * 10, sizeof(float)); // Use calloc for zero-initialization

    // Directly write labels into the transposed array in column-major order
    for (int i = 0; i < currentBatchSize; i++) {
        for (int j = 0; j < 10; j++) {
            int idx = (startLabelIndex + i) * 10 + j; // Calculate the global index in the labels array
            // Calculate the index in the transposed array for column-major order
            int transposedIdx = j * batchSize + i;
            transposedBatchLabels[transposedIdx] = labels[idx];
        }
    }

    // Note: There might be extra zero-padding for batches that are not full, which should be handled appropriately.

    return transposedBatchLabels; // Return the pointer to the transposed labels for this batch
}

void shuffleDataset(float* images, float* labels, int numberOfImages, int imageSize, int labelSize) {
    for (int i = numberOfImages - 1; i > 0; i--) {
        int j = rand() % (i + 1); // Random index from 0 to i

        // Swap images
        for (int k = 0; k < imageSize; k++) {
            float temp = images[i * imageSize + k];
            images[i * imageSize + k] = images[j * imageSize + k];
            images[j * imageSize + k] = temp;
        }

        // Swap labels
        for (int k = 0; k < labelSize; k++) {
            float temp = labels[i * labelSize + k];
            labels[i * labelSize + k] = labels[j * labelSize + k];
            labels[j * labelSize + k] = temp;
        }
    }
}

int calculate_accuracy(nn* network, float* testImages, float* testLabels, int numberOfTestImages, int inputSize, int nb) {
    int correct = 0; // Count of correct predictions
    int totalBatches = (numberOfTestImages + nb - 1) / nb; // Calculate the total number of batches
    int num_l = network->num_layers - 1; // Index of the last layer

    for (int batchNum = 0; batchNum < totalBatches; batchNum++) {
        // Calculate the current batch size, adjusting for the last batch if necessary
        int currentBatchSize = (batchNum == totalBatches - 1) ? 
                                (numberOfTestImages - batchNum * nb) : nb;

        // Process the images for the current batch
        float* batchImages = processBatch(testImages, numberOfTestImages, inputSize, batchNum, nb, currentBatchSize);
        // Process the labels for the current batch
        float* batchLabels = processLabelsBatch(testLabels, numberOfTestImages, batchNum, nb, currentBatchSize);

        initializeBatch(network);

        // Perform forward propagation on the processed batch of images
        forwardPass(network, batchImages);

        for (int img = 0; img < currentBatchSize; img++) {
            float max = -INFINITY;
            int predict = -1;
            for (int n = 0; n < network->output_size; n++) {
                float activation = network->layers[num_l]->a[n * currentBatchSize + img];
                if (activation > max) {
                    max = activation;
                    predict = n;
                }
            }

            // Determine the actual label by finding the position of the max element in the one-hot encoded vector
            int actual = -1;
            for (int n = 0; n < network->output_size; n++) {
                if (batchLabels[n * currentBatchSize + img] == 1.0) {
                    actual = n;
                    break;
                }
            }

            if (predict == actual) {
                correct++;
            }
        }

        // Free the allocated memory for the batch images and labels after use
        free(batchImages);
        free(batchLabels);
    }

    // Compute and return accuracy as a percentage
    // float accuracy = (float)correct / numberOfTestImages * 100.0;
    // printf("Accuracy: %.2f%%\n", accuracy);

    return correct;
}

void saveValidationLossToFile(FILE *file, float validationLoss) {
    if (file == NULL) {
        fprintf(stderr, "File pointer is NULL\n");
        return;
    }
    
    // Write the validation loss to the file
    fprintf(file, "%f\n", validationLoss);
    // Note: We do not close the file here anymore
}






int main(int argc, char *argv[]) {

    if (argc != 2) {
        printf("Usage: %s <nb>\n", argv[0]);
        return 1;
    }
    
    // Seed the random number generator
    srand(time(NULL));

    // Parse command line arguments
    int nl = 1;
    int nh = 800;
    int ne = 50;
    float alpha = 0.1;

    // Parse command line argument for nb
    int nb = atoi(argv[1]); // Convert the argument to an integer


    int inputSize = 784;
    int outputSize = 10;
    int numLayers = nl + 2; 
    //create array of layer sizes, and have numLayers (not dynamic)
    int layerSizes[numLayers];
    // first layer
    layerSizes[0] = inputSize; 
    // last layer
    layerSizes[nl + 1] = outputSize; 

    // initialize layer size
    for (int l = 1; l <= nl; l++) { 
        layerSizes[l] = nh; 
    }



    nn *neuralNetwork = createNetwork(inputSize, outputSize, nb, numLayers, layerSizes);
    initializeNetwork(neuralNetwork);

    // float cost; int accuracy;
    // int train_size = train_data->size;

     // Load MNIST data
    int numberOfImages;
    float *images = readMNISTImages("train-images-idx3-ubyte", &numberOfImages, &inputSize);
    int numberOfLabels;
    float *labels = readMNISTLabels("train-labels-idx1-ubyte", &numberOfLabels);

    int numberOfTrainingImages = 50000;
    int numberOfValidationImages = 10000; // Assuming you have 60k images in total


    // int totalBatches = (numberOfImages + nb - 1) / nb; // Calculate the total number of batches

     FILE *file = fopen("validation_loss.txt", "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening file\n");
        exit(1);
    }

    startTrainingTime = clock();



    // for each epoch
    for (int epoch = 0; epoch <ne; epoch++){
        
        printf("Training Epoch %d/%d\n", epoch + 1, ne);
        shuffleDataset(images, labels, numberOfImages, inputSize, 10);

        int trainingBatches = (numberOfTrainingImages + nb - 1) / nb; // Calculate the number of batches for training


        // JUST UPDATE THE LAYERS' BATCH SIZE FOR THE LAST LAYER WHEN RUNNING THE TRAINING LOOP IN THE MAIN FUNCTION
        // OR JUST HAVE THE FORWARD PASS ETC TAKE A FIELD THAT IS BATCH SIZE RATHER THAN USING N LAYERS BATCHSIZE
        for (int batchNum = 0; batchNum < trainingBatches; batchNum++) {
           
            // Calculate the current batch size, adjusting for the last batch if necessary
            int currentBatchSize = (batchNum == trainingBatches - 1) ? 
                                    (numberOfTrainingImages - batchNum * nb) : nb;

            
            // printf("Entering function X\n");

            float* batchImages = processBatch(images, numberOfTrainingImages, inputSize, batchNum, nb, currentBatchSize);
            
            // Process the labels for the current batch
            float* batchLabels = processLabelsBatch(labels, numberOfTrainingImages, batchNum, nb, currentBatchSize);

           
            // INIT BATCH (ZERO OUT THE GRADIENTS)
             
            initializeBatch(neuralNetwork);
             // Check if this is the last batch

            train_batch(batchImages, batchLabels, currentBatchSize, neuralNetwork);
            
             
            // update_nn(network, alpha, currentBatchSize);
            update_network(neuralNetwork, alpha, currentBatchSize);

            // Free the allocated memory for the batch labels after use
            free(batchLabels);
        }

        // Validation Set 
        float validationLoss = 0.0;
        int validationBatches = (numberOfValidationImages + nb - 1) / nb; // Calculate the number of batches for validation
        for (int batchNum = 0; batchNum < validationBatches; batchNum++) {
            int currentBatchSize = (batchNum == validationBatches - 1) ?
                               (numberOfValidationImages - batchNum * nb) : nb;
            
             // Process the images for the current validation batch
            float* batchImages = processBatch(images + numberOfTrainingImages * inputSize, // Offset to start from the 50k-th image
                                            numberOfValidationImages, inputSize, batchNum, nb, currentBatchSize);

            // Process the labels for the current validation batch
            float* batchLabels = processLabelsBatch(labels + numberOfTrainingImages * 10, // Offset to start from the 50k-th label
                                                    numberOfValidationImages, batchNum, nb, currentBatchSize);


            // Perform forward propagation on the processed batch of validation images
            initializeBatch(neuralNetwork); // Make sure you zero out any previous state
            forwardPass(neuralNetwork, batchImages);

            // Compute the loss on the validation set
            validationLoss += cross_entropy_loss(neuralNetwork, batchLabels, currentBatchSize);


             // Free the allocated memory for the batch images and labels after use
        free(batchImages);
        free(batchLabels);
        }


        validationLoss /= validationBatches; // Average the loss over all validation batches
        printf("Validation Loss after Epoch %d: %f\n", epoch + 1, validationLoss);

        saveValidationLossToFile(file, validationLoss);
    }


    endTrainingTime = clock();
    fclose(file);
    float trainingDuration = (float)(endTrainingTime - startTrainingTime) / CLOCKS_PER_SEC;
    grindRate = (numberOfImages * ne) / trainingDuration; // ne is the number of epochs

    int numberOfTestImages, testImageSize;
    float *testImages = readMNISTImages("t10k-images-idx3", &numberOfTestImages, &testImageSize);
    int numberOfTestLabels;
    float *testLabels = readMNISTLabels("t10k-labels", &numberOfTestLabels);


    startInferenceTime = clock();
    int correct_predictions = calculate_accuracy(neuralNetwork, testImages, testLabels, numberOfTestImages, inputSize, nb);
    endInferenceTime = clock();
    float inferenceDuration = (float)(endInferenceTime - startInferenceTime) / CLOCKS_PER_SEC;

    // Optionally, calculate and print accuracy percentage
    float accuracy = (float)correct_predictions / numberOfTestImages * 100.0;
    printf("Test Set Accuracy: %.2f%%\n", accuracy);


    // Output the performance metrics
    printf("Training Duration: %.2f seconds\n", trainingDuration);
    printf("Grind Rate: %.2f samples/second\n", grindRate);
    printf("Inference Duration: %.2f seconds\n", inferenceDuration);

    free(images);
    free(labels);
    free_nn(neuralNetwork);
}
