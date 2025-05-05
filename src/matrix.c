#include "matrix.h"

#include <stdlib.h>
#include <stdio.h>

Matrix createMatrix(int rows, int cols, float array[][cols]) {
    Matrix matrix;
    matrix.values = (float**) malloc(rows * sizeof(float*));
    matrix.rows = rows;
    matrix.cols = cols;

    int i, j;
    for (i = 0; i < rows; i++) {
        matrix.values[i] = (float*) malloc(cols * sizeof(float));
        for (j = 0; j < cols; j++) {
            matrix.values[i][j] = array[i][j];
        }
    }

    return matrix;
}

Matrix createEmptyMatrix(int rows, int cols) {
    Matrix matrix;
    matrix.values = (float**) malloc(rows * sizeof(float*));
    matrix.rows = rows;
    matrix.cols = cols;

    int i, j;
    for (i = 0; i < rows; i++) {
        matrix.values[i] = (float*) malloc(cols * sizeof(float));
        for (j = 0; j < cols; j++) {
            matrix.values[i][j] = 0.0f;
        }
    }

    return matrix;
}

Matrix createOnesMatrix(int rows, int cols) {
    Matrix matrix;
    matrix.values = (float**) malloc(rows * sizeof(float*));
    matrix.rows = rows;
    matrix.cols = cols;

    int i, j;
    for (i = 0; i < rows; i++) {
        matrix.values[i] = (float*) malloc(cols * sizeof(float));
        for (j = 0; j < cols; j++) {
            matrix.values[i][j] = 1.0f;
        }
    }

    return matrix;
}

Matrix createIdentityMatrix(int size) {
    Matrix result = createEmptyMatrix(size, size);

    int i, j;
    for (i = 0; i < result.rows; i++) {
        for (j = 0; j < result.cols; j++) {
            if (i == j) {
                result.values[i][j] = 1.0f;
            }
        }
    }

    return result;
}

Matrix addMatrices(Matrix matrix1, Matrix matrix2) {
    Matrix result = createEmptyMatrix(matrix1.rows, matrix1.cols);

    int i, j;
    for (i = 0; i < matrix1.rows; i++) {
        for (j = 0; j < matrix1.cols; j++) {
            result.values[i][j] = matrix1.values[i][j] + matrix2.values[i][j];
        }
    }

    return result;
}

Matrix addMatricesWithRepeat(Matrix matrix, Matrix repeatMatrix, int dim) {
    Matrix result = createEmptyMatrix(matrix.rows, matrix.cols);

    if (dim == 0) {
        int i, j;
        for (i = 0; i < matrix.rows; i++) {
            for (j = 0; j < matrix.cols; j++) {
                result.values[i][j] = matrix.values[i][j] + repeatMatrix.values[0][j];
            }
        }
    }
    else {
        int i, j;
        for (i = 0; i < matrix.rows; i++) {
            for (j = 0; j < matrix.cols; j++) {
                result.values[i][j] = matrix.values[i][j] + repeatMatrix.values[i][0];
            }
        }
    }

    return result;
}

Matrix addMatrixWithScalar(Matrix matrix, float scalar) {
    Matrix result = createEmptyMatrix(matrix.rows, matrix.cols);

    int i, j;
    for (i = 0; i < matrix.rows; i++) {
        for (j = 0; j < matrix.cols; j++) {
            result.values[i][j] = matrix.values[i][j] + scalar;
        }
    }

    return result;
}

Matrix multiplyMatrices(Matrix matrix1, Matrix matrix2) {
    Matrix result = createEmptyMatrix(matrix1.rows, matrix2.cols);

    int i, j, k;
    for (i = 0; i < result.rows; i++) {
        for (j = 0; j < result.cols; j++) {
            for (k = 0; k < matrix1.cols; k++) {
                result.values[i][j] += matrix1.values[i][k] * matrix2.values[k][j];
            }
        }
    }

    return result;
}

Matrix multiplyMatricesElementWise(Matrix matrix1, Matrix matrix2) {
    Matrix result = createEmptyMatrix(matrix1.rows, matrix1.cols);

    int i, j;
    for (i = 0; i < matrix1.rows; i ++) {
        for (j = 0; j < matrix1.cols; j++) {
            result.values[i][j] =  matrix1.values[i][j] * matrix2.values[i][j];
        }
    }

    return result;
}

Matrix multiplyMatrixWithScalar(Matrix matrix, float scalar) {
    Matrix result = createEmptyMatrix(matrix.rows, matrix.cols);

    int i, j;
    for (i = 0; i < matrix.rows; i ++) {
        for (j = 0; j < matrix.cols; j++) {
            result.values[i][j] =  matrix.values[i][j] * scalar;
        }
    }

    return result;
}

Matrix indexMatrix(int startRows, int endRows, int startCols, int endCols, Matrix matrix) {
    Matrix result = createEmptyMatrix(endRows - startRows, endCols - startCols);

    int i, j;
    for (i = 0; i < result.rows; i++) {
        for (j = 0; j < result.cols; j++) {
            result.values[i][j] = matrix.values[i + startRows][j + startCols];
        }
    }

    return result;
}

Matrix editMatrix(Matrix matrix, Matrix change, int startRow, int startCol) {
    int i, j;
    for (i = 0; i < change.rows; i++) {
        for (j = 0; j < change.cols; j++) {
            matrix.values[i + startRow][j+startCol] = change.values[i][j];
        }
    }

    return matrix;
}

Matrix combineMatrices(Matrix matrix1, Matrix matrix2, int axis) {
    if (axis == 0) {
        Matrix result = createEmptyMatrix(matrix1.rows + matrix2.rows, matrix1.cols);

        int i, j;
        for (i = 0; i < matrix1.rows; i++) {
            for (j = 0; j < result.cols; j++) {
                result.values[i][j] = matrix1.values[i][j];
            } 
        }
        for (i = 0; i < matrix2.rows; i++) {
            for (j = 0; j < result.cols; j++) {
                result.values[i + matrix1.rows][j] = matrix2.values[i][j];
            }
        }

        return result;
    }

    Matrix result = createEmptyMatrix(matrix1.rows, matrix1.cols + matrix2.cols);

    int i, j;
    for (i = 0; i < result.rows; i++) {
        for (j = 0; j < matrix1.cols; j++) {
            result.values[i][j] = matrix1.values[i][j];
        }
        for (j = 0; j < matrix2.cols; j++) {
            result.values[i][j + matrix1.cols] = matrix2.values[i][j];
        }
    }

    return result;
}

float determinantOfMatrix(Matrix matrix) {
    if (matrix.rows == 1)
    {
        return matrix.values[0][0];
    }
    
    float result = 0;
    int factor = 1;

    int i;
    for (i = 0; i < matrix.cols; i++) {
        float multiplier = matrix.values[0][i] * factor;

        Matrix left = indexMatrix(1, matrix.rows, 0, i, matrix);
        Matrix right = indexMatrix(1, matrix.rows, i + 1, matrix.cols, matrix);
        Matrix combined = combineMatrices(left, right, 1);
        float value = determinantOfMatrix(combined);

        result += multiplier * value;
        factor *= -1;
    }

    return result;
}

Matrix transposeMatrix(Matrix matrix) {
    Matrix result = createEmptyMatrix(matrix.cols, matrix.rows);

    int i, j;
    for (i = 0; i < result.rows; i++) {
        for (j = 0; j < result.cols; j++) {
            result.values[i][j] = matrix.values[j][i];
        }
    }

    return result;
}

Matrix copyMatrix(Matrix matrix) {
    Matrix result = createEmptyMatrix(matrix.rows, matrix.cols);

    int i, j;
    for (i = 0; i < matrix.rows; i++) {
        for (j = 0; j < matrix.cols; j++) {
            result.values[i][j] = matrix.values[i][j];
        }
    }

    return result;
}

void printMatrix(Matrix matrix) {
    int i, j;
    printf("[\n");
    for (i = 0; i < matrix.rows; i++) {
        for (j = 0; j < matrix.cols; j++) {
            if (j == matrix.cols - 1) {
                printf("%f", matrix.values[i][j]);
                continue;
            }
            printf("%f, ", matrix.values[i][j]);
        }
        printf("\n");
    }
    printf("]\n");
}

void freeMatrix(Matrix matrix) {
    int i;
    for (i = 0; i < matrix.rows; i++) {
        free(matrix.values[i]);
    }
    free(matrix.values);
}

void freeMatrices(Matrix* matrices, int n) {
    int i;
    for (i = 0; i < n; i++) {
        freeMatrix(matrices[i]);
    }
    free(matrices);
}