#ifndef STORAGE_H_
#define STORAGE_H_

#include "matrix.h"

typedef struct {
    Matrix* initialObservations;
    Matrix* nextObservations;
    int* actions;
    float* rewards;
    int* dones;
    int size;
    int current;
} Storage;

typedef struct {
    Matrix* initialObservations;
    Matrix* nextObservations;
    int* actions;
    float* rewards;
    int* dones;
    int size;
} Batch;

Storage createStorage(int size, int obsRows, int obsCols);
void storeInStorage(Storage* storage, Matrix currentObservation, Matrix nextObservation, int action, float reward, int done);
Batch sampleBatch(Storage storage, int batchSize);
int* random_subset(int start_idx, int end_idx, int size);
void freeStorage(Storage storage);
void freeBatch(Batch batch);

#endif