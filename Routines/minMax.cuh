#include "common.cuh"

void computeMinMax(float* input, float* output, int num_elements)
{
    int s = -1;
    findBlockSize(&s, &num_elements);

    int block_size = powf(2, s - 1) * blockSize1;
    int num_blocks = num_elements / block_size;
    int remainder  = num_elements - num_blocks * block_size;
    int start_addr = num_elements - remainder;

    if(s == 1)
        find_min_max<blockSize1, threads><<<num_blocks, threads>>>(input, output);
    else if(s == 2)
        find_min_max<blockSize1 * 2, threads><<<num_blocks, threads>>>(input, output);
    else if(s == 3)
        find_min_max<blockSize1 * 4, threads><<<num_blocks, threads>>>(input, output);
    else if(s == 4)
        find_min_max<blockSize1 * 8, threads><<<num_blocks, threads>>>(input, output);
    else
        find_min_max<blockSize1 * 16, threads><<<num_blocks, threads>>>(input, output);

    find_min_max_dynamic<threads><<<1, threads>>>(input, output, num_elements, start_addr, num_blocks);
}


unsigned long long int getMinMax(float* cpu_in, float* input, float* output, int num_elements, float* cpu_out)
{
	unsigned long long int start;
	unsigned long long int delta;

    cudaMalloc(&input, num_elements * sizeof(float));
    cudaMalloc(&output, num_elements * sizeof(float));

    cudaMemcpy(input, cpu_in, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    
    start = get_clock();
    computeMinMax(input, output, num_elements);
    cudaDeviceSynchronize();
    delta = get_clock() - start;

    cudaMemcpy(cpu_out, output, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    return delta;
}
