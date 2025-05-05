#include "matrix.h"
#include "neural.h"
#include "activation.h"
#include "loss.h"
#include "rl.h"
#include "optimizer.h"
#include "storage.h"
#include "cartpole.h"
#include "video.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int main() {
    // Training parameters
    float learningRate = 0.0001f;
    float epsilonMax = 1.0f;
    float epsilonMin = 0.05f;
    float epsilonDecay = 1.0f / 4000.0f;
    float gamma = 0.99f;
    int targetUpdateInterval = 100;
    int storageSize = 1000;
    int batchSize = 32;
    int numSteps = 20000;

    srand(time(NULL));

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
            printf("Step: %d, Episode reward: %f, Action0: %d, Action1: %d\n", i + 1, episodeReward, episodeAction0Taken, episodeAction1Taken);
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

    // Record 3 episodes
    episodeReward = 0.0f;
    for (i = 0; i < 3; i++) {
        cartPoleState = resetCartPole();

        episodeReward = 0.0f;

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

        int done = 0;
        
        Matrix videoFrames[200];
        int frameCount = 0;
        while (done == 0)
        {
            Matrix* output = forward(state, network);
            float action0Value = output[network.length - 1].values[0][0];
            float action1Value = output[network.length - 1].values[0][1];
            int action = (action0Value > action1Value) ? 0 : 1; // argmax
            freeMatrices(output, network.length);

            cartPoleState = stepCartPole(action, cartPoleState);
            done = cartPoleState.terminated;
            episodeReward += cartPoleState.reward;
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
            videoFrames[frameCount] = renderCartPole(cartPoleState);
            frameCount++;
        }
        printf("Video Episode Reward: %f, Frame Count: %d\n", episodeReward, frameCount);

        // file name with index
        char filename[50];
        snprintf(filename, sizeof(filename), "cartpole_episode_%d.y4m", i);
        saveMatricesToY4M(videoFrames, frameCount, 30, filename);
        printf("Video saved to %s\n", filename);
        
        for (int j = 0; j < frameCount; j++) {
            freeMatrix(videoFrames[j]);
        }

        freeMatrix(state);
    }

    // Free resources
    freeNeural(network);
    freeNeural(targetNetwork);
    freeOptimizer(optimizer);
    freeStorage(storage);

    printf("Training completed.\n");

    return 0;
}