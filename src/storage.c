#include "storage.h"
#include "matrix.h"

#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>

Storage createStorage(int size, int obsRows, int obsCols) {
    Storage storage;
    storage.initialObservations = (Matrix*) malloc(size * sizeof(Matrix));
    storage.nextObservations = (Matrix*) malloc(size * sizeof(Matrix));
    storage.actions = (int*) malloc(size * sizeof(int));
    storage.rewards = (float*) malloc(size * sizeof(float));
    storage.dones = (int*) malloc(size * sizeof(int));
    storage.size = size;
    storage.current = 0;

    int i;
    for (i = 0; i < size; i ++) {
        storage.initialObservations[i] = createEmptyMatrix(obsRows, obsCols);
        storage.nextObservations[i] = createEmptyMatrix(obsRows, obsCols);
    }

    return storage;
}

void storeInStorage(Storage* storage, Matrix currentObservation, Matrix nextObservation, int action, float reward, int done) {
    Matrix initialCopy = copyMatrix(currentObservation);
    Matrix nextCopy = copyMatrix(nextObservation);
    
    freeMatrix(storage->initialObservations[storage->current]);
    freeMatrix(storage->nextObservations[storage->current]);
    
    storage->initialObservations[storage->current] = initialCopy;
    storage->nextObservations[storage->current] = nextCopy;
    storage->actions[storage->current] = action;
    storage->rewards[storage->current] = reward;
    storage->dones[storage->current] = done;

    if (storage->current < (storage->size - 1)) {
        storage->current += 1;
    } else {
        storage->current = 0;
    }
}

int* random_subset(int start_idx, int end_idx, int size) {
    int range_size = end_idx - start_idx + 1;
    
    int *indices = (int *)malloc(range_size * sizeof(int));
    int *result = (int *)malloc(size * sizeof(int));
    
    for (int i = 0; i < range_size; i++) {
        indices[i] = start_idx + i;
    }
    
    for (int i = 0; i < size; i++) {
        int j = i + rand() % (range_size - i);
        
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
        
        result[i] = indices[i];
    }
    
    free(indices);
    return result;
}

Batch sampleBatch(Storage storage, int batchSize) {
    Batch batch;
    batch.size = batchSize;
    batch.initialObservations = (Matrix*) malloc(batchSize * sizeof(Matrix));
    batch.nextObservations = (Matrix*) malloc(batchSize * sizeof(Matrix));
    batch.actions = (int*) malloc(batchSize * sizeof(int));
    batch.rewards = (float*) malloc(batchSize * sizeof(float));
    batch.dones = (int*) malloc(batchSize * sizeof(int));
    
    int* indices = random_subset(0, storage.size - 1, batchSize);
    
    int i;
    for (i = 0; i < batchSize; i++) {
        batch.initialObservations[i] = copyMatrix(storage.initialObservations[indices[i]]);
        batch.nextObservations[i] = copyMatrix(storage.nextObservations[indices[i]]);
        batch.actions[i] = storage.actions[indices[i]];
        batch.rewards[i] = storage.rewards[indices[i]];
        batch.dones[i] = storage.dones[indices[i]];
    }

    free(indices);

    return batch;
}

void freeStorage(Storage storage) {
    int i;
    for (i = 0; i < storage.size; i++) {
        freeMatrix(storage.initialObservations[i]);
        freeMatrix(storage.nextObservations[i]);
    }
    free(storage.initialObservations);
    free(storage.nextObservations);
    free(storage.actions);
    free(storage.rewards);
    free(storage.dones);
}

void freeBatch(Batch batch) {
    int i;
    for (i = 0; i < batch.size; i++) {
        freeMatrix(batch.initialObservations[i]);
        freeMatrix(batch.nextObservations[i]);
    }
    free(batch.initialObservations);
    free(batch.nextObservations);
    free(batch.actions);
    free(batch.rewards);
    free(batch.dones);
}