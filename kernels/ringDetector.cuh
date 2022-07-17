#ifndef RINGDETECTOR_CUH
#define RINGDETECTOR_CUH

#include <cuda_profiler_api.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "util.cuh"

#define TILE_SIZE 32
#define SIZE 1024


__global__ void rgb2HSV(uint8_t* r, uint8_t* g, uint8_t* b, uint8_t* h, uint8_t* s, uint8_t* v, int num_rows, int num_cols);

__global__ void createHSVMask(uint8_t* h, uint8_t* s, uint8_t* v, int num_cols, int num_rows, uint8_t* hsv_mask, 
                        uint8_t _h_ml1, uint8_t _s_ml1, uint8_t _v_ml1, uint8_t _h_mh1, uint8_t _s_mh1, uint8_t _v_mh1, 
                        uint8_t _h_ml2, uint8_t _s_ml2, uint8_t _v_ml2, uint8_t _h_mh2, uint8_t _s_mh2, uint8_t _v_mh2);

__global__ void _fisheye_bitwise_and(uint8_t* hsv_mask, uint8_t* fisheye_mask, int num_cols);

__device__ inline float getMedian(float* array);
#endif