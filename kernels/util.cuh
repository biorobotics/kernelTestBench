#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <npp.h>
#include <nppi_morphological_operations.h>
#include <nppdefs.h>

#include <iostream>
#include <vector>

#define MAX_CUDA_THREADS_PER_BLOCK 1024

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void findBlockSize(int* whichSize, int* num_el);


__device__ inline float getMedian(float* array);