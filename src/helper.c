#include "helper.h"

#include <stdlib.h>
#include <string.h>
#include <stddef.h>

const char* getStrArg(int argc, char* argv[], const char* key, const char* def) {
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], key) == 0) return argv[i + 1];
    }
    return def;
}
int getIntArg(int argc, char* argv[], const char* key, int def) {
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], key) == 0) return atoi(argv[i + 1]);
    }
    return def;
}

float getFloatArg(int argc, char* argv[], const char* key, float def) {
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], key) == 0) return (float)atof(argv[i + 1]);
    }
    return def;
}