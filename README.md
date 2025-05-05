# DQN from Scratch in C

This repository contains a complete implementation of a Deep Q-Network (DQN) reinforcement learning algorithm from scratch in pure C, with no external dependencies. The project includes a custom implementation of the CartPole environment, neural network architecture, matrix operations, and visualisation capabilities.

![CartPole Demo](example.gif)

## Project Overview

The entire project was built from the ground up to test and strengthen my proficiency in deep learning algorithms and reinforcement learning concepts.

## Features Implemented

### Deep Learning Components
- **Neural Network Framework**: Complete feedforward neural network with customisable layers
- **Matrix Operations**: Matrix operations
- **Activation Functions**: ReLU implementation
- **Loss Functions**: Huber loss
- **Adam Optimizer**: Advanced adaptive optimisation algorithm

### Reinforcement Learning Components
- **Experience Replay Buffer**: Stores and samples past experiences
- **Target Network**: Stabilises learning by slowly updating target Q-values
- **Epsilon-Greedy Policy**: Balances exploration and exploitation
- **Q-Learning**: Implements discounted rewards for long-term planning

### Environment and Visualization
- **CartPole Environment**: Classic control problem implementation
- **Simple Graphics System**: Visualises the CartPole state
- **Video Recording**: Capability to save agent episodes as Y4M videos

## Compiling and Execution

This project uses no third-party libraries, so all you need is a GCC compiler.

To compile the project, run the following command from the project root:

```bash
gcc -Iinclude train.c src/*.c -o train -lm
```

To run the program after compilation:

```bash
./train
```

When you run the program, it will:
1. Initialise the neural network and environment
2. Train the DQN agent for a specified number of steps
3. Record 3 episodes of the trained agent
4. Save the recordings as Y4M video files

## Customisation

You can adjust the training parameters in `train.c`, including:
- Learning rate
- Epsilon decay rate (controls exploration vs. exploitation)
- Discount factor
- Network architecture
- Number of training steps
- Batch size
- Experience replay buffer size

Feel free to experiment with different parameters to optimise training for the CartPole task.

## Contact

If you have any questions about this project or are interested in discussing deep learning research opportunities, academic collaborations, or potential roles please reach out to me at tristanjlegg@gmail.com.

This project demonstrates my deep understanding of neural networks and reinforcement learning by implementing everything from scratch with no dependencies.
