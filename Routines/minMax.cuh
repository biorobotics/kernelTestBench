#ifndef MINMAX_CUH
#define MINMAX_CUH


#include "common.cuh"

void computeMinMax(float* input, float* output, int num_elements);
unsigned long long int getMinMax(float* cpu_in, float* input, float* output, int num_elements, float* cpu_out);

#endif