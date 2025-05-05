#include "activation.h"

#include <stdio.h>
#include <math.h>

Matrix activation(Matrix input, Activation activation) {
    // Create a copy of the input to avoid modifying it
    Matrix result = copyMatrix(input);
    
    if (activation == ReLU) {
        int i, j;
        for (i = 0; i < result.rows; i++) {
            for (j = 0; j < result.cols; j++) {
                result.values[i][j] = (float) fmax(0, result.values[i][j]);
            }
        }
    } else if (activation == Tanh) {
        int i, j;
        for (i = 0; i < result.rows; i++) {
            for (j = 0; j < result.cols; j++) {
                result.values[i][j] = (float) tanh(result.values[i][j]);
            }
        }
    }
    return result;
}

Matrix activationDerivative(Matrix output, Activation activation) {
    Matrix derivative = createEmptyMatrix(output.rows, output.cols);

    int i, j;
    for (i = 0; i < output.rows; i++) {
        for (j = 0; j < output.cols; j++) {
            if (activation == ReLU) {
                // ReLU: 1 if x > 0, 0 otherwise
                derivative.values[i][j] = output.values[i][j] > 0 ? 1.0f : 0.0f;
            } else if (activation == Tanh) {
                // Tanh: 1 - tanh²(x)
                float tanhVal = output.values[i][j];
                derivative.values[i][j] = 1.0f - (tanhVal * tanhVal);
            } else {
                // None: 1
                derivative.values[i][j] = 1.0f;
            }
        }
    }
    
    return derivative;
}

void printActivations(Activation* activations, int length) {
    printf("[");
    int i;
    for (i = 0; i < length; i++) {
        if (activations[i] == ReLU) {
            if (i == length - 1) {
                printf("ReLU");
            } else {
                printf("ReLU, ");
            }
        } else if (activations[i] == Tanh) {
            if (i == length - 1) {
                printf("Tanh");
            } else {
                printf("Tanh, ");
            }
        } else if (activations[i] == None) {
            if (i == length - 1) {
                printf("None");
            } else {
                printf("None, ");
            }
        }
    }
    printf("]\n");
}
