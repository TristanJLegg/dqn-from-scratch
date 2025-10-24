#ifndef HELPER_H_
#define HELPER_H_

#include <stddef.h>

const char* getStrArg(int argc, char* argv[], const char* key, const char* def);
int getIntArg(int argc, char* argv[], const char* key, int def);
float getFloatArg(int argc, char* argv[], const char* key, float def);

#endif