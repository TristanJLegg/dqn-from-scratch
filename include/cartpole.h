#ifndef CARTPOLE_H_
#define CARTPOLE_H_

#include "matrix.h"

typedef struct {
    double cartPosition;
    double cartVelocity;
    double poleAngle;
    double poleAngularVelocity;
    float reward;
    int terminated;
    int step;
} CartPoleState;

CartPoleState resetCartPole();
CartPoleState stepCartPole(int action, CartPoleState cartPoleState);
Matrix renderCartPole(CartPoleState cartPoleState);
void printCartPoleState(CartPoleState cartPoleState);

#endif