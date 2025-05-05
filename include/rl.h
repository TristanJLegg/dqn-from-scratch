#ifndef RL_H_
#define RL_H_

#include "neural.h"
#include "loss.h"
#include "storage.h"

float calculateTarget(Matrix nextState, float reward, int done, Neural targetNetwork, float gamma);

#endif