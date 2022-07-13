#include <cuda.h>
#include "reduction.cuh"


__device__ void warp_reduce_max( volatile float smem[64])
{

	smem[threadIdx.x] = smem[threadIdx.x+32] > smem[threadIdx.x] ? 
						smem[threadIdx.x+32] : smem[threadIdx.x]; DEBUG_SYNC;

	smem[threadIdx.x] = smem[threadIdx.x+16] > smem[threadIdx.x] ? 
						smem[threadIdx.x+16] : smem[threadIdx.x]; DEBUG_SYNC;

	smem[threadIdx.x] = smem[threadIdx.x+8] > smem[threadIdx.x] ? 
						smem[threadIdx.x+8 ] : smem[threadIdx.x]; DEBUG_SYNC;

	smem[threadIdx.x] = smem[threadIdx.x+4] > smem[threadIdx.x] ? 
						smem[threadIdx.x+4] : smem[threadIdx.x]; DEBUG_SYNC;

	smem[threadIdx.x] = smem[threadIdx.x+2] > smem[threadIdx.x] ? 
						smem[threadIdx.x+2] : smem[threadIdx.x]; DEBUG_SYNC;

	smem[threadIdx.x] = smem[threadIdx.x+1] > smem[threadIdx.x] ? 
						smem[threadIdx.x+1] : smem[threadIdx.x]; DEBUG_SYNC;

}

__device__ void warp_reduce_min(volatile float smem[64])
{

	smem[threadIdx.x] = smem[threadIdx.x+32] < smem[threadIdx.x] ? 
						smem[threadIdx.x+32] : smem[threadIdx.x]; DEBUG_SYNC;

	smem[threadIdx.x] = smem[threadIdx.x+16] < smem[threadIdx.x] ? 
						smem[threadIdx.x+16] : smem[threadIdx.x]; DEBUG_SYNC;

	smem[threadIdx.x] = smem[threadIdx.x+8] < smem[threadIdx.x] ? 
						smem[threadIdx.x+8] : smem[threadIdx.x]; DEBUG_SYNC;

	smem[threadIdx.x] = smem[threadIdx.x+4] < smem[threadIdx.x] ? 
						smem[threadIdx.x+4] : smem[threadIdx.x]; DEBUG_SYNC;

	smem[threadIdx.x] = smem[threadIdx.x+2] < smem[threadIdx.x] ? 
						smem[threadIdx.x+2] : smem[threadIdx.x]; DEBUG_SYNC;

	smem[threadIdx.x] = smem[threadIdx.x+1] < smem[threadIdx.x] ? 
						smem[threadIdx.x+1] : smem[threadIdx.x]; DEBUG_SYNC;

}

template<int threads>
__global__ void find_min_max_dynamic(float* in, float* out, int n, int start_adr, int num_blocks)
{

	__shared__ float smem_min[64];
	__shared__ float smem_max[64];

	int tid = threadIdx.x + start_adr;

	float max = -inf;
	float min = inf;
	float val;


	// tail part
	int mult = 0;
	for(int i = 1; mult + tid < n; i++)
	{
		val = in[tid + mult];
	
		min = val < min ? val : min;
		max = val > max ? val : max;

		mult = int_mult(i,threads);
	}

	// previously reduced MIN part
	mult = 0;
	int i;
	for(i = 1; mult+threadIdx.x < num_blocks; i++)
	{
		val = out[threadIdx.x + mult];

		min = val < min ? val : min;
		
		mult = int_mult(i,threads);
	}

	// MAX part
	for(; mult+threadIdx.x < num_blocks*2; i++)
	{
		val = out[threadIdx.x + mult];

		max = val > max ? val : max;
		
		mult = int_mult(i,threads);
	}


	if(threads == 32)
	{
		smem_min[threadIdx.x+32] = 0.0f;
		smem_max[threadIdx.x+32] = 0.0f;

	}
	
	smem_min[threadIdx.x] = min;
	smem_max[threadIdx.x] = max;

	__syncthreads();

	if(threadIdx.x < 32)
	{
		warp_reduce_min(smem_min);
		warp_reduce_max(smem_max);
	}
	if(threadIdx.x == 0)
	{
		out[blockIdx.x] = smem_min[threadIdx.x]; // out[0] == ans
		out[blockIdx.x + gridDim.x] = smem_max[threadIdx.x]; 
	}


}

template<int els_per_block, int threads>
__global__ void find_min_max(float* in, float* out)
{
	__shared__ float smem_min[64];
	__shared__ float smem_max[64];

	int tid = threadIdx.x + blockIdx.x*els_per_block;

	float max = -inf;
	float min = inf;
	float val;

	const int iters = els_per_block/threads;
	
#pragma unroll
		for(int i = 0; i < iters; i++)
		{

			val = in[tid + i*threads];

			min = val < min ? val : min;
			max = val > max ? val : max;

		}
	
	
	if(threads == 32)
	{
		smem_min[threadIdx.x+32] = 0.0f;
		smem_max[threadIdx.x+32] = 0.0f;
	
	}
	
	smem_min[threadIdx.x] = min;
	smem_max[threadIdx.x] = max;


	__syncthreads();

	if(threadIdx.x < 32)
	{
		warp_reduce_min(smem_min);
		warp_reduce_max(smem_max);
	}
	if(threadIdx.x == 0)
	{
		out[blockIdx.x] = smem_min[threadIdx.x]; // out[0] == ans
		out[blockIdx.x + gridDim.x] = smem_max[threadIdx.x]; 
	}

}
