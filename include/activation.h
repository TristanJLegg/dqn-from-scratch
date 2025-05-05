#ifndef ACTIVATION_H_
#define ACTIVATION_H_

#include "matrix.h"

typedef enum {
    ReLU,
    Tanh,
    None
} Activation;

Matrix activation(Matrix input, Activation activation);
Matrix activationDerivative(Matrix output, Activation activation);
void printActivations(Activation* activations, int length);

#endif