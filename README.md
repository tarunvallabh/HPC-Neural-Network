# Neural Network for MNIST Digit Classification

This project implements a simple neural network in C for classifying handwritten digits from the MNIST dataset. The neural network is trained using stochastic gradient descent.

## Prerequisites

- GCC compiler for compiling C code.
- NVCC compiler for CUDA code (if applicable).
- OpenBLAS for optimized basic linear algebra operations.

Ensure that OpenBLAS is installed on your system. If you are using CUDA features, make sure CUDA is installed and configured properly.

## Compilation

A Makefile is provided for easy compilation of the CPU code in the `src` directory. Use the following command in the terminal:

```
make
```


This will compile the code and generate an executable named `final`.

For compiling the CUDA version, navigate to the `cuda` subdirectory within the `src` directory: 
```
cd cuda
```
A seperate Makefile is provided for easy compilation of the CUDA code as well. Use the following command in the terminal: 

```
make
```
This will compile the code and generate an executable named `rawcuda`.


## Running the Program

After compilation, you can run the CPU program by specifying the number of training samples per batch (`nb`) as a command-line argument:

```
./final <nb>
```

You can run the CUDA program also by specifying the number of training samples per batch (`nb`) as a command-line argument:

```
./rawcuda <nb>
```

## Data

The program expects the MNIST dataset to be present in the same directory with the following filenames:

- Training images: `train-images-idx3-ubyte`
- Training labels: `train-labels-idx1-ubyte`
- Test images: `t10k-images-idx3`
- Test labels: `t10k-labels`


## Output

The program will train the neural network on the MNIST training dataset and then evaluate its performance on the test dataset. The output includes the accuracy of the model on the test set, the training duration, grind rate (samples/second), and inference duration. It will also compute the average cross entropy loss and save it to the `validation_loss.txt` file. To further plot the data from the text file, run the following command: 
```
python plotting.py
```