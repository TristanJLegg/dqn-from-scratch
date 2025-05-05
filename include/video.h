#ifndef VIDEO_H_
#define VIDEO_H_

#include "matrix.h"

void drawRectangle(float value, int* pointsX, int* pointsY, int numPoints, Matrix* image);
void drawCircle(float value, int centerX, int centerY, int radius, Matrix* image);
void drawLine(float value, int position, int axis, Matrix* image);
int saveMatrixAsBMP(Matrix matrix, const char* filename);
int saveMatricesToY4M(Matrix* frames, int frameCount, int fps, const char* filename);

#endif