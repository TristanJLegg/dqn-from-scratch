# Minimum CMake version
cmake_minimum_required(VERSION 3.10)

# Project name
project(DQN C)

# Add source files
add_executable(train
    train.c
    matrix.c
    neural.c
)