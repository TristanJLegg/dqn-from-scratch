/*
    This is based off the python implementation at https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
*/

#include "cartpole.h"
#include "video.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define PI 3.14159265358979323846

void rotateRad(float x, float y, float angle, float* resultX, float* resultY) {
    float cos_angle = cosf(angle);
    float sin_angle = sinf(angle);
    
    *resultX = x * cos_angle - y * sin_angle;
    *resultY = x * sin_angle + y * cos_angle;
}

CartPoleState resetCartPole() {
    CartPoleState cartPoleState;
    cartPoleState.cartPosition = ((double) rand() / RAND_MAX) * 0.1 - 0.05;
    cartPoleState.cartVelocity = ((double) rand() / RAND_MAX) * 0.1 - 0.05;
    cartPoleState.poleAngle = ((double) rand() / RAND_MAX) * 0.1 - 0.05;
    cartPoleState.poleAngularVelocity = ((double) rand() / RAND_MAX) * 0.1 - 0.05;
    cartPoleState.reward = 0;
    cartPoleState.terminated = 0;
    cartPoleState.step = 0;

    return cartPoleState;
}

CartPoleState stepCartPole(int action, CartPoleState cartPoleState) {
    // Return same game state if already terminated
    if (cartPoleState.terminated == 1) {
        return cartPoleState;
    }

    double gravity = 9.8;
    double massCart = 1.0;
    double massPole = 0.1;
    double totalMass = massCart + massPole;
    double length = 0.5; // half length of the pole
    double poleMassLength = massPole * length;
    double forceMagnitude = 10.0;
    double tau = 0.02; // seconds between state updates

    double thetaThresholdRadians = 12 * 2 * PI / 360;
    double xThreshold = 2.4;
    double force = action == 1 ? forceMagnitude : -forceMagnitude;
    double x = cartPoleState.cartPosition;
    double xDot = cartPoleState.cartVelocity;
    double theta = cartPoleState.poleAngle;
    double thetaDot = cartPoleState.poleAngularVelocity;

    double temp = (force + poleMassLength * thetaDot * thetaDot * sin(theta)) / totalMass;
    double thetaAcc = (gravity * sin(theta) - cos(theta) * temp) / (length * ((4.0 / 3.0) - ((massPole * cos(theta) * cos(theta) / totalMass))));
    double xAcc = temp - (poleMassLength * thetaAcc * cos(theta) / totalMass);

    x += tau * xDot;
    xDot += tau * xAcc;
    theta += tau * thetaDot;
    thetaDot += tau * thetaAcc;

    cartPoleState.cartPosition = x;
    cartPoleState.cartVelocity = xDot;
    cartPoleState.poleAngle = theta;
    cartPoleState.poleAngularVelocity = thetaDot;
    
    cartPoleState.step += 1;

    if (x < -xThreshold || x > xThreshold || theta < -thetaThresholdRadians || theta > thetaThresholdRadians || cartPoleState.step >= 200) {
        cartPoleState.terminated = 1;
    } else {
        cartPoleState.terminated = 0;
    }
    cartPoleState.reward = 1;

    return cartPoleState;
}

Matrix renderCartPole(CartPoleState cartPoleState) {
    double worldWidth = 4.8;
    double scale = 600 / worldWidth;
    double poleWidth = 10.0;
    double poleLen = scale;
    double cartWidth = 50.0;
    double cartHeight = 30.0;

    Matrix base = createOnesMatrix(400, 600);
    Matrix image = multiplyMatrixWithScalar(base, 255.0f);
    freeMatrix(base);

    double l = -cartWidth / 2;
    double r = cartWidth / 2;
    double t = cartHeight / 2;
    double b = -cartHeight / 2;
    double axleOffset = cartHeight / 4.0f;

    // Cart
    double cartX = cartPoleState.cartPosition * scale + 300;
    double cartY = 300;
    int cartPointsXArr[] = {l + cartX, r+ cartX, r + cartX, l + cartX};
    int cartPointsYArr[] = {b + cartY, b + cartY, t + cartY, t + cartY};
    int* cartPointsX = cartPointsXArr;
    int* cartPointsY = cartPointsYArr;
    int numCartPoints = 4;
    drawRectangle(0.0f, cartPointsX, cartPointsY, numCartPoints, &image);

    // Pole
    l = -poleWidth / 2;
    r = poleWidth / 2;
    t = poleLen - (poleWidth / 2);
    b = -poleWidth / 2;

    double polePointsXRelative[] = {l, r, r, l};
    double polePointsYRelative[] = {b, b, t, t};
    int* polePointsX = (int*) malloc(4 * sizeof(int));
    int* polePointsY = (int*) malloc(4 * sizeof(int));

    int i;
    for (i = 0; i < 4; i++) {
        float resultX, resultY;
        rotateRad(polePointsXRelative[i], polePointsYRelative[i], cartPoleState.poleAngle, &resultX, &resultY);
        polePointsX[i] = (int) (resultX + cartX);
        polePointsY[i] = -resultY - axleOffset + cartY;
    }
    drawRectangle(150.0f, polePointsX, polePointsY, 4, &image);

    drawCircle(75.0f, cartX, cartY - axleOffset, poleWidth / 2, &image);

    drawLine(0.0f, cartY, 0, &image);

    free(polePointsX);
    free(polePointsY);

    return image;
}

void printCartPoleState(CartPoleState cartPoleState) {
    printf("CartPoleState: {\n");
    printf("    x: %f\n", cartPoleState.cartPosition);
    printf("    xDot: %f\n", cartPoleState.cartVelocity);
    printf("    theta: %f\n", cartPoleState.poleAngle);
    printf("    thetaDot: %f\n", cartPoleState.poleAngularVelocity);
    printf("    terminated: %d\n", cartPoleState.terminated);
    printf("    reward: %f\n", cartPoleState.reward);
    printf("}\n");
}