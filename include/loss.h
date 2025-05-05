#ifndef LOSS_H_
#define LOSS_H_

#include "matrix.h"
#include "storage.h"

typedef struct {
    float value;
    Matrix outputGradient;
    Matrix target;
    Matrix prediction;
} Loss;

Loss calculateLoss(Matrix prediction, Matrix target, float delta);
void freeLoss(Loss loss);

#endif