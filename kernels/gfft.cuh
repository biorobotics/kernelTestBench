#ifndef GFFT_CUH
#define GFFT_CUH

#include <cuda.h>
#include <cuda_runtime_api.h>

#define BLOCK_SIZE 32
#define PARALLEL_REDUCTION_THREADS 1024

__global__ void sobel(float* x, float* dx, float* dy, int num_cols);
__global__ void getScores(float* dx, float* dy, float* R, int num_cols);
__global__ void filterScores(float* R, float thresh, int num_cols);

#endif