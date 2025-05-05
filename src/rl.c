#include "rl.h"

#include <stdlib.h>

float calculateTarget(Matrix nextState, float reward, int done, Neural targetNetwork, float gamma) {
    Matrix* nextOutputs = forward(nextState, targetNetwork);
    Matrix nextQValues = nextOutputs[targetNetwork.length - 1];
    
    float maxNextQ = nextQValues.values[0][0];
    int j;
    for (j = 1; j < nextQValues.cols; j++) {
        if (nextQValues.values[0][j] > maxNextQ) {
            maxNextQ = nextQValues.values[0][j];
        }
    }
    
    float target = reward + gamma * maxNextQ * (1.0f - (float)done);
    
    freeMatrices(nextOutputs, targetNetwork.length);
    
    return target;
}