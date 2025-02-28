// In build directory run:
// cmake .. && make && ./train && rm ./train

#include "matrix.h"
#include "neural.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    srand(time(NULL));

    int neurons[] = {2, 2};
    Neural neural = createNeuralWithRandomWeights(2, neurons);
    printf("Weights and Biases:\n");
    printNeural(neural);

    float inputArray[1][2] = {{0.6, -0.4}};
    Matrix input = createMatrix(1, 2, inputArray);
    printf("Input:\n");
    printMatrix(input);

    Matrix result = forward(input, neural);
    printf("Output:\n");
    printMatrix(result);

    return 0;
}