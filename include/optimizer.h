
#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include "neural.h"
#include "matrix.h"

// Adam optimizer struct
typedef struct {
    Matrix** m; // First moment vectors (for weights and biases)
    Matrix** v; // Second moment vectors (for weights and biases)
    float beta1;
    float beta2;
    float epsilon;
    int t; // Timestep counter
    int length; // Number of layers
} AdamOptimizer;

// Create a new Adam optimizer for a neural network
AdamOptimizer createOptimizer(Neural network, float beta1, float beta2, float epsilon);

// Update neural network parameters using Adam optimizer
void updateNeural(Neural* neural, Gradients gradients, float learningRate, AdamOptimizer* optimizer);

// Free resources used by the optimizer
void freeOptimizer(AdamOptimizer optimizer);

#endif