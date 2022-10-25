#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

int main()
{
    cudaDeviceProp dP;
    float minComputeCapability = 3.0;

    int rc = cudaGetDeviceProperties(&dP, 0);
    if(rc != cudaSuccess) {
        cudaError_t error = cudaGetLastError();
        printf("CUDA error: %s", cudaGetErrorString(error));
        return rc;
    }
    if((dP.major+(dP.minor/10)) < minComputeCapability) {
        printf("Min Compute Capability of %2.1f required:  %d.%d found\n Not Building CUDA Code", minComputeCapability, dP.major, dP.minor);
        return 1; /* Failure */
    } else {
        printf("%d%d", dP.major, dP.minor);
        return 0; /* Success */
    }
}
