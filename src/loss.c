#include "loss.h"
#include "neural.h"

#include <math.h>

// Huber Loss
Loss calculateLoss(Matrix prediction, Matrix target, float delta) {
    Loss loss;
    loss.value = 0.0f;
    loss.outputGradient = createEmptyMatrix(prediction.rows, prediction.cols);
    loss.target = copyMatrix(target);
    loss.prediction = copyMatrix(prediction);

    float sumError = 0.0f;
    int numElements = prediction.rows * prediction.cols;
    
    int i, j;
    for (i = 0; i < prediction.rows; i++) {
        for (j = 0; j < prediction.cols; j++) {
            float error = prediction.values[i][j] - target.values[i][j];
            float absError = fabs(error);

            if (absError <= delta) {
                sumError += 0.5f * error * error;
                loss.outputGradient.values[i][j] = error;
            } else {
                sumError += delta * (absError - 0.5f * delta);
                loss.outputGradient.values[i][j] = (error > 0 ? delta : -delta);
            }
        }
    }

    loss.value = sumError / numElements;

    return loss;
}

void freeLoss(Loss loss) {
    freeMatrix(loss.outputGradient);
    freeMatrix(loss.target);
    freeMatrix(loss.prediction);
}