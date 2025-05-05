#include "optimizer.h"

#include <stdlib.h>
#include <math.h>

AdamOptimizer createOptimizer(Neural network, float beta1, float beta2, float epsilon) {
    AdamOptimizer optimizer;
    optimizer.beta1 = beta1;
    optimizer.beta2 = beta2;
    optimizer.epsilon = epsilon;
    optimizer.t = 0;
    optimizer.length = network.length - 1;
    
    // Allocate memory for first and second moment vectors
    optimizer.m = (Matrix**)malloc(optimizer.length * sizeof(Matrix*));
    optimizer.v = (Matrix**)malloc(optimizer.length * sizeof(Matrix*));
    
    // Initialize moment matrices for each layer
    int i;
    for (i = 0; i < optimizer.length; i++) {
        optimizer.m[i] = (Matrix*)malloc(2 * sizeof(Matrix)); // For weights and biases
        optimizer.v[i] = (Matrix*)malloc(2 * sizeof(Matrix));
        
        // For weights
        optimizer.m[i][0] = createEmptyMatrix(network.weights[i].rows, network.weights[i].cols);
        optimizer.v[i][0] = createEmptyMatrix(network.weights[i].rows, network.weights[i].cols);
        
        // For biases
        optimizer.m[i][1] = createEmptyMatrix(1, network.biases[i].cols);
        optimizer.v[i][1] = createEmptyMatrix(1, network.biases[i].cols);
    }
    
    return optimizer;
}

void updateNeural(Neural* neural, Gradients gradients, float learningRate, AdamOptimizer* optimizer) {
    optimizer->t++;
    
    float alphaT = learningRate * sqrt(1.0f - pow(optimizer->beta2, optimizer->t)) / 
                  (1.0f - pow(optimizer->beta1, optimizer->t));
    
    int i, j, k;
    for (i = 0; i < optimizer->length; i++) {
        for (j = 0; j < neural->weights[i].rows; j++) {
            for (k = 0; k < neural->weights[i].cols; k++) {
                float gradient = gradients.dWeights[i].values[j][k];
                
                optimizer->m[i][0].values[j][k] = optimizer->beta1 * optimizer->m[i][0].values[j][k] + 
                                             (1.0f - optimizer->beta1) * gradient;
                
                optimizer->v[i][0].values[j][k] = optimizer->beta2 * optimizer->v[i][0].values[j][k] + 
                                             (1.0f - optimizer->beta2) * gradient * gradient;
                
                neural->weights[i].values[j][k] -= alphaT * optimizer->m[i][0].values[j][k] / 
                                               (sqrt(optimizer->v[i][0].values[j][k]) + optimizer->epsilon);
            }
        }
        
        for (k = 0; k < neural->biases[i].cols; k++) {
            float gradient = gradients.dBiases[i].values[0][k];
            
            optimizer->m[i][1].values[0][k] = optimizer->beta1 * optimizer->m[i][1].values[0][k] + 
                                         (1.0f - optimizer->beta1) * gradient;
            
            optimizer->v[i][1].values[0][k] = optimizer->beta2 * optimizer->v[i][1].values[0][k] + 
                                         (1.0f - optimizer->beta2) * gradient * gradient;
            
            neural->biases[i].values[0][k] -= alphaT * optimizer->m[i][1].values[0][k] / 
                                          (sqrt(optimizer->v[i][1].values[0][k]) + optimizer->epsilon);
        }
    }
}

void freeOptimizer(AdamOptimizer optimizer) {
    int i;
    for (i = 0; i < optimizer.length; i++) {
        freeMatrix(optimizer.m[i][0]);
        freeMatrix(optimizer.v[i][0]);
        
        freeMatrix(optimizer.m[i][1]);
        freeMatrix(optimizer.v[i][1]);
        
        free(optimizer.m[i]);
        free(optimizer.v[i]);
    }
    
    free(optimizer.m);
    free(optimizer.v);
}