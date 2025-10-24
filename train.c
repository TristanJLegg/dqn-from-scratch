#include "matrix.h"
#include "neural.h"
#include "activation.h"
#include "loss.h"
#include "rl.h"
#include "optimizer.h"
#include "storage.h"
#include "cartpole.h"
#include "video.h"
#include "helper.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

int main(int argc, char* argv[]) {
    // Training parameters
    float learningRate       = (float) getFloatArg(argc, argv, "--lr", 0.0001);
    float epsilonMax         = (float) getFloatArg(argc, argv, "--eps-max", 1.0);
    float epsilonMin         = (float) getFloatArg(argc, argv, "--eps-min", 0.05);
    float epsilonDecay       = (float) getFloatArg(argc, argv, "--eps-decay", 1.0 / 4000.0);
    float gamma              = (float) getFloatArg(argc, argv, "--gamma", 0.99);
    int targetUpdateInterval = (int) getIntArg(argc, argv, "--target-update", 100);
    int storageSize          = (int) getIntArg(argc, argv, "--storage", 1000);
    int batchSize            = (int) getIntArg(argc, argv, "--batch", 32);
    int numSteps             = (int) getIntArg(argc, argv, "--steps", 20000);

    srand(time(NULL));

    printf("Training with parameters:\n");
    printf("  Learning Rate       = %f\n", learningRate);
    printf("  Epsilon Max         = %f\n", epsilonMax);
    printf("  Epsilon Min         = %f\n", epsilonMin);
    printf("  Epsilon Decay       = %f\n", epsilonDecay);
    printf("  Gamma               = %f\n", gamma);
    printf("  Target Update Intvl = %d\n", targetUpdateInterval);
    printf("  Storage Size        = %d\n", storageSize);
    printf("  Batch Size          = %d\n", batchSize);
    printf("  Num Steps           = %d\n", numSteps);


    // Initialize the network and the target network
    printf("Initializing neural networks...\n");
    int neurons[] = {4, 128, 128, 2}; // 4 inputs (state), 128 hidden neurons, 128 hidden neurons, 1 output (action)
    Activation activations[] = {ReLU, ReLU, None};
    Neural network = createNeuralWithRandomWeights(4, neurons, activations);
    Neural targetNetwork = createNeuralWithRandomWeights(4, neurons, activations);

    // Initialize the Adam optimizer
    printf("Initializing optimizer...\n");
    AdamOptimizer optimizer = createOptimizer(network, 0.9f, 0.999f, 1e-8f);

    // Initialize the environment
    printf("Initializing environment...\n");
    CartPoleState cartPoleState = resetCartPole();

    // Initialize the storage
    printf("Initializing storage...\n");
    Storage storage = createStorage(storageSize, 1, 4);

    // Initial state
    Matrix state = createMatrix(
        1,
        4,
        (float[][4]){{
            cartPoleState.cartPosition, 
            cartPoleState.cartVelocity, 
            cartPoleState.poleAngle, 
            cartPoleState.poleAngularVelocity
        }}
    );

    // Initializations
    int updateCount = 0;
    float epsilon = epsilonMax;

    // Logging
    float episodeReward = 0.0f;
    int episodeAction0Taken = 0;
    int episodeAction1Taken = 0;

    // Training loop
    printf("Starting training...\n");
    int i;
    for (i = 0; i < numSteps; i++) {
        int action;
        if (((float)rand() / (float)RAND_MAX) < epsilon) {
            action = rand() % 2;
        } else {
            Matrix* output = forward(state, network);
            float action0Value = output[network.length - 1].values[0][0];
            float action1Value = output[network.length - 1].values[0][1];
            action = (action0Value > action1Value) ? 0 : 1; // argmax
            freeMatrices(output, network.length);
        }

        if (action == 0) {
            episodeAction0Taken++;
        } else {
            episodeAction1Taken++;
        }
        
        cartPoleState = stepCartPole(action, cartPoleState);
        episodeReward += cartPoleState.reward;

        Matrix nextState = createMatrix(
            1,
            4,
            (float[][4]){{
                cartPoleState.cartPosition, 
                cartPoleState.cartVelocity, 
                cartPoleState.poleAngle, 
                cartPoleState.poleAngularVelocity
            }}
        );

        storeInStorage(&storage, state, nextState, action, cartPoleState.reward, cartPoleState.terminated);

        // Sample from storage and train the network
        if (i >= storageSize) {
            Batch batch = sampleBatch(storage, batchSize);
            
            int j;
            for (j = 0; j < batch.size; j++) {
                Matrix observation = batch.initialObservations[j];
                Matrix nextObservation = batch.nextObservations[j];

                Matrix* outputs = forward(observation, network);
                
                float targetValue = calculateTarget(
                    nextObservation, 
                    batch.rewards[j],
                    batch.dones[j],
                    targetNetwork,
                    gamma
                );

                Matrix currentQValues = copyMatrix(outputs[network.length - 1]);
                Matrix targetQValues = copyMatrix(currentQValues);

                int actionIndex = batch.actions[j];
                targetQValues.values[0][actionIndex] = targetValue;

                Loss loss = calculateLoss(currentQValues, targetQValues, 1.0);
                
                Gradients gradients = backward(observation, outputs, network, loss);
                freeLoss(loss);         
                updateNeural(&network, gradients, learningRate, &optimizer);
                freeGradients(gradients, network.length - 1);

                freeMatrices(outputs, network.length);
                
                freeMatrix(currentQValues);
                freeMatrix(targetQValues);
            }
            
            epsilon = fmax(epsilon - ((epsilonMax - epsilonMin) * epsilonDecay), epsilonMin);
            
            updateCount++;
            if (updateCount % targetUpdateInterval == 0) {
                freeNeural(targetNetwork);
                targetNetwork = copyNeural(network);
            }

            freeBatch(batch);
        }

        // Reset the environment if done
        if (cartPoleState.terminated) {
            cartPoleState = resetCartPole();
            printf("Step: %-6d  Reward: %-10.2f  Action0: %-6d  Action1: %-6d\n", i + 1, episodeReward, episodeAction0Taken, episodeAction1Taken);

            episodeReward = 0.0f;
            episodeAction0Taken = 0;
            episodeAction1Taken = 0;
            freeMatrix(state);
            state = createMatrix(
                1,
                4,
                (float[][4]){{
                    cartPoleState.cartPosition, 
                    cartPoleState.cartVelocity, 
                    cartPoleState.poleAngle, 
                    cartPoleState.poleAngularVelocity
                }}
            );
        } else {
            freeMatrix(state);
            state = nextState;
        }
    }

    // Save the neural net
    save(&network, "model.bin");

    // Free resources
    freeNeural(network);
    freeNeural(targetNetwork);
    freeOptimizer(optimizer);
    freeStorage(storage);

    printf("Training completed.\n");

    return 0;
}