#ifndef MATRIX_H_
#define MATRIX_H_

typedef struct {
    float** values;
    int rows;
    int cols;
} Matrix;

Matrix createMatrix(int rows, int cols, float array[][cols]);
Matrix createEmptyMatrix(int rows, int cols);
Matrix createOnesMatrix(int rows, int cols);
Matrix createIdentityMatrix(int size);
Matrix addMatrices(Matrix matrix1, Matrix matrix2);
Matrix addMatricesWithRepeat(Matrix matrix, Matrix repeatMatrix, int dim);
Matrix addMatrixWithScalar(Matrix matrix, float scalar);
Matrix multiplyMatrices(Matrix matrix1, Matrix matrix2);
Matrix multiplyMatricesElementWise(Matrix matrix1, Matrix matrix2);
Matrix multiplyMatrixWithScalar(Matrix matrix, float scalar);
Matrix indexMatrix(int startRows, int endRows, int startCols, int endCols, Matrix Matrix);
Matrix editMatrix(Matrix matrix, Matrix change, int row, int col);
Matrix combineMatrices(Matrix matrix1, Matrix matrix2, int axis);
float determinantOfMatrix(Matrix matrix);
Matrix transposeMatrix(Matrix matrix);
Matrix copyMatrix(Matrix matrix);
void printMatrix(Matrix matrix);
void freeMatrix(Matrix matrix);
void freeMatrices(Matrix* matrices, int n);

#endif