#include "matrix.h"
#include "neural.h"
#include "activation.h"
#include "cartpole.h"
#include "video.h"
#include "helper.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char* argv[]) {
    const char* modelPath = getStrArg(argc, argv, "--model", "model.bin");
    const char* outVideo  = getStrArg(argc, argv, "--out", "episode.y4m");
    const int fps = 30;

    printf("Loading model: %s\n", modelPath);

    Neural network = (Neural){0};
    load(&network, modelPath);

    srand((unsigned)time(NULL));

    CartPoleState stateEnv = resetCartPole();

    Matrix state = createMatrix(
        1, 4,
        (float[][4]){{
            stateEnv.cartPosition,
            stateEnv.cartVelocity,
            stateEnv.poleAngle,
            stateEnv.poleAngularVelocity
        }}
    );

    int capacity = 256;
    int count = 0;
    Matrix* frames = (Matrix*)malloc(sizeof(Matrix) * capacity);

    int done = 0;
    float episodeReward = 0.0f;

    while (!done) {
        Matrix* outputs = forward(state, network);
        float q0 = outputs[network.length - 1].values[0][0];
        float q1 = outputs[network.length - 1].values[0][1];
        int action = (q0 > q1) ? 0 : 1;
        freeMatrices(outputs, network.length);

        stateEnv = stepCartPole(action, stateEnv);
        episodeReward += stateEnv.reward;
        done = stateEnv.terminated;

        freeMatrix(state);
        state = createMatrix(
            1, 4,
            (float[][4]){{
                stateEnv.cartPosition,
                stateEnv.cartVelocity,
                stateEnv.poleAngle,
                stateEnv.poleAngularVelocity
            }}
        );

        if (count >= capacity) {
            capacity *= 2;
            frames = (Matrix*)realloc(frames, sizeof(Matrix) * capacity);
        }
        frames[count++] = renderCartPole(stateEnv);
    }

    printf("Episode reward: %f, frames: %d\n", episodeReward, count);

    saveMatricesToY4M(frames, count, fps, outVideo);
    printf("Saved video to %s\n", outVideo);

    for (int i = 0; i < count; ++i) {
        freeMatrix(frames[i]);
    }
    free(frames);

    freeMatrix(state);
    freeNeural(network);

    return 0;
}
