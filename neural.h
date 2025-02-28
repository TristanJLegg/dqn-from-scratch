#ifndef NEURAL_H_
#define NEURAL_H_

#include "matrix.h"

typedef struct {
    Matrix* weights;
    Matrix* biases;
    int length;
} Neural;

Neural createNeuralWithZeroWeights(int length, int neurons[]);
Neural createNeuralWithRandomWeights(int length, int neurons[]);
Matrix forward(Matrix input, Neural neural);
void printNeural(Neural neural);

#endif