#include "neural.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

Neural createNeuralWithZeroWeights(int length, int neurons[], Activation activations[]) {
    Neural neural;
    neural.weights = (Matrix*) malloc((length - 1) * sizeof(Matrix));
    neural.biases = (Matrix*) malloc((length - 1) * sizeof(Matrix));
    neural.neurons = (int*) malloc(length * sizeof(int));
    neural.activations = (Activation*) malloc((length - 1) * sizeof(Activation));
    neural.length = length;

    int i;
    for (i = 0; i < length - 1; i++) {
        neural.weights[i] = createEmptyMatrix(neurons[i], neurons[i+1]);
        neural.biases[i] = createEmptyMatrix(1, neurons[i+1]);
        neural.neurons[i] = neurons[i];
        neural.activations[i] = activations[i];
    }
    neural.neurons[length - 1] = neurons[length - 1];

    return neural;
}

Neural createNeuralWithRandomWeights(int length, int neurons[], Activation activations[]) {
    Neural neural;
    neural.weights = (Matrix*) malloc((length - 1) * sizeof(Matrix));
    neural.biases = (Matrix*) malloc((length - 1) * sizeof(Matrix));
    neural.neurons = (int*) malloc(length * sizeof(int));
    neural.activations = (Activation*) malloc((length - 1) * sizeof(Activation));
    neural.length = length;

    int i, j, k;
    for (i = 0; i < length - 1; i++) {
        neural.weights[i] = createEmptyMatrix(neurons[i], neurons[i+1]);
        neural.biases[i] = createEmptyMatrix(1, neurons[i+1]);
        neural.neurons[i] = neurons[i];
        neural.activations[i] = activations[i];

        // Initialize with Xavier/Glorot initialization
        float scale = sqrt(2.0f / (neurons[i] + neurons[i+1]));
        for (j = 0; j < neural.weights[i].rows; j++) {
            for (k = 0; k < neural.weights[i].cols; k++) {
                // Random between -scale and +scale
                neural.weights[i].values[j][k] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
            }
        }
        
        // Initialize biases to a small constant (0.01)
        for (j = 0; j < neural.biases[i].rows; j++) {
            for (k = 0; k < neural.biases[i].cols; k++) {
                neural.biases[i].values[j][k] = 0.01f;
            }
        }
    }
    neural.neurons[length - 1] = neurons[length - 1];

    return neural;
}

Matrix* forward(Matrix input, Neural neural) {
    Matrix* outputs = malloc(neural.length * sizeof(Matrix));
    
    // Make a copy of the input
    Matrix currentInput = copyMatrix(input);
    
    // Process each layer
    for (int i = 0; i < neural.length - 1; i++) {
        // Layer operation: output = input * weight + bias
        Matrix weightedInput = multiplyMatrices(currentInput, neural.weights[i]);
        Matrix layerOutput = addMatricesWithRepeat(weightedInput, neural.biases[i], 0);
        freeMatrix(weightedInput);
        
        // Apply activation function
        Matrix activated = activation(layerOutput, neural.activations[i]);
        freeMatrix(layerOutput);
        
        // Store output for this layer
        outputs[i] = activated;
        
        // Prepare for next layer
        if (i < neural.length - 2) {
            Matrix nextInput = copyMatrix(activated);
            freeMatrix(currentInput); // Free the current input
            currentInput = nextInput;
        } else {
            // On the last layer, we'll use activated as the final output
            outputs[neural.length - 1] = copyMatrix(activated);
            freeMatrix(currentInput);
        }
    }
    
    return outputs;
}

Gradients backward(Matrix input, Matrix* layerOutputs, Neural neural, Loss loss) {
    // Allocate memory for gradients
    Gradients gradients;
    gradients.dWeights = (Matrix*) malloc((neural.length - 1) * sizeof(Matrix));
    gradients.dBiases = (Matrix*) malloc((neural.length - 1) * sizeof(Matrix));
    
    // Initialize gradient matrices
    for (int i = 0; i < neural.length - 1; i++) {
        gradients.dWeights[i] = createEmptyMatrix(neural.weights[i].rows, neural.weights[i].cols);
        gradients.dBiases[i] = createEmptyMatrix(neural.biases[i].rows, neural.biases[i].cols);
    }
    
    // Start with the output gradient from the loss function
    Matrix delta = copyMatrix(loss.outputGradient);
    
    // Loop backwards through the layers
    for (int i = neural.length - 2; i >= 0; i--) {
        // Current layer output (or input for first layer)
        Matrix currentOutput = (i == 0) ? input : layerOutputs[i - 1];
        
        // Input to current layer's activation function
        Matrix activationInput;
        if (i == neural.length - 2) {
            // For the last layer, we use the actual output
            activationInput = layerOutputs[i];
        } else {
            // For hidden layers, we need to compute it
            Matrix weightedInput = multiplyMatrices(currentOutput, neural.weights[i]);
            activationInput = addMatricesWithRepeat(weightedInput, neural.biases[i], 0);
            freeMatrix(weightedInput);
        }
        
        // Compute activation derivative
        Matrix activationDeriv = activationDerivative(activationInput, neural.activations[i]);
        
        // Apply chain rule: delta = delta * activation_derivative
        Matrix oldDelta = delta;
        delta = multiplyMatricesElementWise(delta, activationDeriv);
        freeMatrix(oldDelta);
        freeMatrix(activationDeriv);
        
        // Compute weight gradients: dW = currentOutput^T * delta
        Matrix currentOutputT = transposeMatrix(currentOutput);
        freeMatrix(gradients.dWeights[i]);
        gradients.dWeights[i] = multiplyMatrices(currentOutputT, delta);
        freeMatrix(currentOutputT);
        
        // Compute bias gradients: db = sum(delta) = delta (for batch size 1)
        freeMatrix(gradients.dBiases[i]);
        gradients.dBiases[i] = copyMatrix(delta);
        
        // Propagate delta to previous layer (except for input layer)
        if (i > 0) {
            Matrix weightT = transposeMatrix(neural.weights[i]);
            Matrix newDelta = multiplyMatrices(delta, weightT);
            freeMatrix(weightT);
            freeMatrix(delta);
            delta = newDelta;
        } else {
            freeMatrix(delta);
        }
        
        // Free intermediate activationInput if we created it
        if (i != neural.length - 2) {
            freeMatrix(activationInput);
        }
    }
    
    return gradients;
}

// Gradients* backward(Matrix input, Matrix* layerOutputs, Neural neural, Loss loss) {
//     // Allocate memory for the gradients struct
//     Gradients* gradients = (Gradients*)malloc(sizeof(Gradients));
    
//     // Initialize fields
//     gradients->dWeights = (Matrix*) malloc((neural.length - 1) * sizeof(Matrix));
//     gradients->dBiases = (Matrix*) malloc((neural.length - 1) * sizeof(Matrix));
    
//     // Initialize gradient matrices
//     for (int i = 0; i < neural.length - 1; i++) {
//         gradients->dWeights[i] = createEmptyMatrix(neural.weights[i].rows, neural.weights[i].cols);
//         gradients->dBiases[i] = createEmptyMatrix(neural.biases[i].rows, neural.biases[i].cols);
//     }
    
//     // Set dInput field to indicate it doesn't contain valid matrix data
//     gradients->dInput.values = NULL;
//     gradients->dInput.rows = 0;
//     gradients->dInput.cols = 0;
    
//     // Start with the output gradient from the loss function
//     Matrix delta = copyMatrix(loss.outputGradient);
    
//     // Loop backwards through the layers
//     for (int i = neural.length - 2; i >= 0; i--) {
//         // Current layer output (or input for first layer)
//         Matrix currentOutput = (i == 0) ? input : layerOutputs[i - 1];
        
//         // Input to current layer's activation function
//         Matrix activationInput;
//         int needToFree = 0;
        
//         if (i == neural.length - 2) {
//             // For the last layer, we use the actual output
//             activationInput = layerOutputs[i];
//             // Do NOT set needToFree = 1 here, as we don't own this memory
//         } else {
//             // For hidden layers, we need to compute it
//             Matrix weightedInput = multiplyMatrices(currentOutput, neural.weights[i]);
//             activationInput = addMatricesWithRepeat(weightedInput, neural.biases[i], 0);
//             freeMatrix(weightedInput);
//             needToFree = 1;  // We created this matrix, so we must free it
//         }
        
//         // Compute activation derivative
//         Matrix activationDeriv = activationDerivative(activationInput, neural.activations[i]);
        
//         // Apply chain rule: delta = delta * activation_derivative
//         Matrix oldDelta = delta;
//         delta = multiplyMatricesElementWise(delta, activationDeriv);
//         freeMatrix(oldDelta);
//         freeMatrix(activationDeriv);
        
//         // Compute weight gradients: dW = currentOutput^T * delta
//         Matrix currentOutputT = transposeMatrix(currentOutput);
//         gradients->dWeights[i] = multiplyMatrices(currentOutputT, delta);
//         freeMatrix(currentOutputT);
        
//         // Compute bias gradients: db = sum(delta) = delta (for batch size 1)
//         gradients->dBiases[i] = copyMatrix(delta);
        
//         // Propagate delta to previous layer (except for input layer)
//         if (i > 0) {
//             Matrix weightT = transposeMatrix(neural.weights[i]);
//             Matrix newDelta = multiplyMatrices(delta, weightT);
//             freeMatrix(weightT);
//             freeMatrix(delta);
//             delta = newDelta;
//         } else {
//             freeMatrix(delta);
//         }
        
//         // Free activationInput ONLY if we created it
//         if (needToFree) {
//             freeMatrix(activationInput);
//         }
//     }
    
//     return gradients;
// }

Neural copyNeural(Neural neural) {
    Neural copy = createNeuralWithZeroWeights(neural.length, neural.neurons, neural.activations);

    int i, j, k;
    for (i = 0; i < neural.length - 1; i++) {
        for (j = 0; j < neural.weights[i].rows; j++) {
            for (k = 0; k < neural.weights[i].cols; k++) {
                copy.weights[i].values[j][k] = neural.weights[i].values[j][k];
            }
        }
        for (j = 0; j < neural.biases[i].rows; j++) {
            for (k = 0; k < neural.biases[i].cols; k++) {
                copy.biases[i].values[j][k] = neural.biases[i].values[j][k];
            }
        }
    }

    return copy;
}

void printNeural(Neural neural) {
    int i;
    // Print weights
    printf("Weights:\n");
    for (i = 0; i < neural.length - 1; i++) {
        printf("%d:\n", i);
        printMatrix(neural.weights[i]);
    }

    printf("Biases:\n");
    for (i = 0; i < neural.length - 1; i++) {
        printf("%d:\n", i);
        printMatrix(neural.biases[i]);
    }

    printf("Activations:\n");
    printActivations(neural.activations, neural.length - 1);
}

void freeNeural(Neural neural) {
    int i;
    for (i = 0; i < neural.length - 1; i++) {
        freeMatrix(neural.weights[i]);
        freeMatrix(neural.biases[i]);
    }
    free(neural.weights);
    free(neural.biases);
    free(neural.neurons);
    free(neural.activations);
}

void freeGradients(Gradients gradients, int n) {
    int i;
    for (i = 0; i < n; i++) {
        freeMatrix(gradients.dWeights[i]);
        freeMatrix(gradients.dBiases[i]);
    }
    
    free(gradients.dWeights);
    free(gradients.dBiases);
}