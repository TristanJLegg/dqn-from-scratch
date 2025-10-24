#ifndef NEURAL_H_
#define NEURAL_H_

#include "matrix.h"
#include "activation.h"
#include "loss.h"

typedef struct {
    Matrix* weights;
    Matrix* biases;
    int* neurons;
    Activation* activations;
    int length;
} Neural;

typedef struct {
    Matrix* dWeights;
    Matrix* dBiases;
    Matrix dInput;
} Gradients;

Neural createNeuralWithZeroWeights(int length, int neurons[], Activation activations[]);
Neural createNeuralWithRandomWeights(int length, int neurons[], Activation activations[]);
Matrix* forward(Matrix input, Neural neural);
Gradients backward(Matrix input, Matrix* layerOutputs, Neural neural, Loss loss);
Neural copyNeural(Neural neural);
void save(const Neural* network, const char* path);
void load(Neural* network, const char* path);
void printNeural(Neural neural);
void freeNeural(Neural neural);
void freeGradients(Gradients gradients, int n);

#endif