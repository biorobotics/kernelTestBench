#ifndef REDUCTION_CUH
#define REDUCTION_CUH

#include <cuda.h>


#if __DEVICE_EMULATION__
#define DEBUG_SYNC __syncthreads();
#else
#define DEBUG_SYNC
#endif

#if (__CUDA_ARCH__ < 200)
#define int_mult(x,y)	__mul24(x,y)	
#else
#define int_mult(x,y)	x*y
#endif

#define inf 0x7f800000 



const int blockSize1 = 4096/2; 
const int threads = 64;


__device__ void warp_reduce_max(volatile float smem[64]);
__device__ void warp_reduce_min(volatile float smem[64]);

template<int threads>
__global__ void find_min_max_dynamic(float* in, float* out, int n, int start_adr, int num_blocks);

template<int els_per_block, int threads>
__global__ void find_min_max(float* in, float* out);


#endif