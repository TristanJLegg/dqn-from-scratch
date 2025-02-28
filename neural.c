#include "neural.h"

#include <stdlib.h>
#include <stdio.h>

Neural createNeuralWithZeroWeights(int length, int neurons[]) {
    Neural neural;
    neural.weights = (Matrix*) malloc((length - 1) * sizeof(Matrix));
    neural.biases = (Matrix*) malloc((length - 1) * sizeof(Matrix));
    neural.length = length;

    int i;
    for (i = 0; i < length - 1; i++) {
        neural.weights[i] = createEmptyMatrix(neurons[i], neurons[i+1]);
        neural.biases[i] = createEmptyMatrix(neurons[i], neurons[i+1]);
    }

    return neural;
}

Neural createNeuralWithRandomWeights(int length, int neurons[]) {
    Neural neural;
    neural.weights = (Matrix*) malloc((length - 1) * sizeof(Matrix));
    neural.biases = (Matrix*) malloc((length - 1) * sizeof(Matrix));
    neural.length = length;

    int i, j, k;
    for (i = 0; i < length - 1; i++) {
        neural.weights[i] = createEmptyMatrix(neurons[i], neurons[i+1]);
        neural.biases[i] = createEmptyMatrix(neurons[i], neurons[i+1]);

        for (j = 0; j < neural.weights[i].rows; j++) {
            for (k = 0; k < neural.weights[i].cols; k++) {
                neural.weights[i].values[j][k] = (((float) rand() / RAND_MAX) * 2) - 1;
            }
        }
    }

    return neural;
}

Matrix forward(Matrix input, Neural neural) {
    int i;
    for (i = 0; i < neural.length - 1; i++) {
        input = addMatrices(multiplyMatrices(input, neural.weights[i]), neural.biases[i]);
    }
    return input;
}

// TODO : Calculate Gradients for backward pass
Matrix* backward(float loss, Neural neural) {
    Matrix* gradients = (Matrix*) malloc((neural.length - 1) * sizeof(Matrix));
    
    int i;
    for (i = 0; i < neural.length - 1; i++) {
        gradients[i] = createEmptyMatrix(neural.weights[i].rows, neural.weights[i].cols);
    }

    return gradients;
}

void updateNeural(Matrix* gradients, Neural neural, float lr) {
    
}

void printNeural(Neural neural) {
    int i, j, k;
    for (i = 0; i < neural.length - 1; i++) {
        printf("%d: [\n", i);
        for (j = 0; j < neural.weights[i].rows; j++) {
            for (k = 0; k < neural.weights[i].cols; k++) {
                if (k == neural.weights[i].cols - 1) {
                    printf("%fx + %f\n", neural.weights[i].values[j][k], neural.biases[i].values[j][k]);
                    continue;
                }
                printf("%fx + %f, ", neural.weights[i].values[j][k], neural.biases[i].values[j][k]);
            }
        }
        printf("]\n");
    }    
}