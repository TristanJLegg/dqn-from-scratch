#include "video.h"
#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void drawRectangle(float value, int* pointsX, int* pointsY, int numPoints, Matrix* image) {    
    int minY = pointsY[0], maxY = pointsY[0];
    for (int i = 1; i < numPoints; i++) {
        if (pointsY[i] < minY) minY = pointsY[i];
        if (pointsY[i] > maxY) maxY = pointsY[i];
    }
    
    minY = (minY < 0) ? 0 : minY;
    maxY = (maxY >= image->rows) ? image->rows - 1 : maxY;
    
    for (int y = minY; y <= maxY; y++) {
        int intersections[numPoints];
        int intersectionCount = 0;
        
        for (int i = 0; i < numPoints; i++) {
            int j = (i + 1) % numPoints;
            
            if ((pointsY[i] <= y && pointsY[j] > y) || 
                (pointsY[j] <= y && pointsY[i] > y)) {
                
                intersections[intersectionCount++] = 
                    pointsX[i] + (y - pointsY[i]) * 
                    (pointsX[j] - pointsX[i]) / 
                    (float)(pointsY[j] - pointsY[i]);
            }
        }
        
        for (int i = 0; i < intersectionCount - 1; i++) {
            for (int j = 0; j < intersectionCount - i - 1; j++) {
                if (intersections[j] > intersections[j + 1]) {
                    int temp = intersections[j];
                    intersections[j] = intersections[j + 1];
                    intersections[j + 1] = temp;
                }
            }
        }
        
        for (int i = 0; i < intersectionCount; i += 2) {
            if (i + 1 < intersectionCount) {
                int startX = (intersections[i] < 0) ? 0 : intersections[i];
                int endX = (intersections[i + 1] >= image->cols) ? image->cols - 1 : intersections[i + 1];
                
                for (int x = startX; x <= endX; x++) {
                    image->values[y][x] = value;
                }
            }
        }
    }
}

void drawCircle(float value, int centerX, int centerY, int radius, Matrix* image) {
    int minX = (centerX - radius < 0) ? 0 : centerX - radius;
    int maxX = (centerX + radius >= image->cols) ? image->cols - 1 : centerX + radius;
    int minY = (centerY - radius < 0) ? 0 : centerY - radius;
    int maxY = (centerY + radius >= image->rows) ? image->rows - 1 : centerY + radius;
    
    for (int y = minY; y <= maxY; y++) {
        for (int x = minX; x <= maxX; x++) {
            int dx = x - centerX;
            int dy = y - centerY;
            int distanceSquared = dx*dx + dy*dy;
            
            if (distanceSquared <= radius*radius) {
                image->values[y][x] = value;
            }
        }
    }
}

void drawLine(float value, int position, int axis, Matrix* image) {    
    if (axis == 0) {
        if (position >= 0 && position < image->rows) {
            for (int x = 0; x < image->cols; x++) {
                image->values[position][x] = value;
            }
        }
    } else {
        if (position >= 0 && position < image->cols) {
            for (int y = 0; y < image->rows; y++) {
                image->values[y][position] = value;
            }
        }
    }
}

int saveMatrixAsBMP(Matrix matrix, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error: Could not open file %s for writing\n", filename);
        return 0;
    }
    
    int padding = (4 - (matrix.cols % 4)) % 4;
    
    int dataSize = (matrix.cols + padding) * matrix.rows;
    int fileSize = 54 + (256 * 4) + dataSize;
    
    unsigned char signature[2] = {'B', 'M'};
    fwrite(signature, 1, 2, file);
    
    fwrite(&fileSize, 4, 1, file);
    
    int reserved = 0;
    fwrite(&reserved, 4, 1, file);
    
    int dataOffset = 54 + (256 * 4);
    fwrite(&dataOffset, 4, 1, file);
    
    int infoHeaderSize = 40;
    fwrite(&infoHeaderSize, 4, 1, file);
    
    fwrite(&matrix.cols, 4, 1, file);
    
    int height = -matrix.rows;
    fwrite(&height, 4, 1, file);
    
    short planes = 1;
    fwrite(&planes, 2, 1, file);
    
    short bitsPerPixel = 8;
    fwrite(&bitsPerPixel, 2, 1, file);
    
    int compression = 0;
    fwrite(&compression, 4, 1, file);
    
    fwrite(&dataSize, 4, 1, file);
    
    int xPixelsPerMeter = 2835;
    fwrite(&xPixelsPerMeter, 4, 1, file);
    
    int yPixelsPerMeter = 2835;
    fwrite(&yPixelsPerMeter, 4, 1, file);
    
    int colorsUsed = 256;
    fwrite(&colorsUsed, 4, 1, file);
    
    int colorsImportant = 256;
    fwrite(&colorsImportant, 4, 1, file);
    
    for (int i = 0; i < 256; i++) {
        unsigned char color[4] = {i, i, i, 0};
        fwrite(color, 1, 4, file);
    }
    
    unsigned char padValue = 0;
    
    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.cols; j++) {
            unsigned char pixel = (unsigned char) matrix.values[i][j];
            fwrite(&pixel, 1, 1, file);
        }
        
        for (int p = 0; p < padding; p++) {
            fwrite(&padValue, 1, 1, file);
        }
    }
    
    fclose(file);
    printf("Successfully saved %dx%d grayscale matrix to %s\n", 
           matrix.cols, matrix.rows, filename);
    return 1;
}

int saveMatricesToY4M(Matrix* frames, int frameCount, int fps, const char* filename) {
    if (frameCount <= 0 || frames == NULL) {
        printf("Error: Invalid frames array\n");
        return 0;
    }
    
    int width = frames[0].cols;
    int height = frames[0].rows;
    
    int frameSize = width * height;
    
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error: Could not open file %s for writing\n", filename);
        return 0;
    }
    
    fprintf(file, "YUV4MPEG2 W%d H%d F%d:1 Ip A1:1 Cmono\n", width, height, fps);
    
    unsigned char* yPlane = (unsigned char*)malloc(frameSize);
    if (!yPlane) {
        fclose(file);
        return 0;
    }
    
    for (int f = 0; f < frameCount; f++) {
        Matrix frame = frames[f];
        
        fprintf(file, "FRAME\n");
        
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                float value = frame.values[i][j];
                
                if (value < 0.0f) value = 0.0f;
                if (value > 255.0f) value = 255.0f;

                yPlane[i * width + j] = (unsigned char)value;
            }
        }
        
        fwrite(yPlane, 1, frameSize, file);
    }
    
    free(yPlane);
    fclose(file);
    
    printf("Successfully saved %d frames (%dx%d) at %d fps to %s\n", frameCount, width, height, fps, filename);
    return 1;
}