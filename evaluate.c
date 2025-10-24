#include "matrix.h"
#include "neural.h"
#include "activation.h"
#include "cartpole.h"
#include "helper.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char* argv[]) {
    const char* modelPath = getStrArg(argc, argv, "--model", "model.bin");
    int episodes = getIntArg(argc, argv, "--episodes", 1);

    printf("Loading model: %s\n", modelPath);
    printf("Running %d evaluation episode(s)\n", episodes);

    Neural network = (Neural){0};
    load(&network, modelPath);

    srand((unsigned)time(NULL));

    float totalReward = 0.0f;

    for (int ep = 0; ep < episodes; ++ep) {
        CartPoleState env = resetCartPole();

        Matrix state = createMatrix(
            1, 4,
            (float[][4]){{
                env.cartPosition,
                env.cartVelocity,
                env.poleAngle,
                env.poleAngularVelocity
            }}
        );

        int done = 0;
        float episodeReward = 0.0f;

        while (!done) {
            Matrix* outputs = forward(state, network);
            float q0 = outputs[network.length - 1].values[0][0];
            float q1 = outputs[network.length - 1].values[0][1];
            int action = (q0 > q1) ? 0 : 1;
            freeMatrices(outputs, network.length);

            env = stepCartPole(action, env);
            episodeReward += env.reward;
            done = env.terminated;

            freeMatrix(state);
            state = createMatrix(
                1, 4,
                (float[][4]){{
                    env.cartPosition,
                    env.cartVelocity,
                    env.poleAngle,
                    env.poleAngularVelocity
                }}
            );
        }

        freeMatrix(state);

        printf("Episode %d reward: %f\n", ep + 1, episodeReward);
        totalReward += episodeReward;
    }

    float avg = totalReward / (float)episodes;
    printf("Average reward over %d episode(s): %f\n", episodes, avg);

    freeNeural(network);
    return 0;
}
