#ifndef TESTMINMAX_H
#define TESTMINMAX_H

#include "common.h"

#define NUM_ITER 250

void testMinMax(uint64_t size, int device_id)
{
    float* d_in, * d_out;
    float* cpu_in, *cpu_out;
    int return_value;
    
    float maxMemory = getTotalMemory(device_id);
    float bandwidth = getPeakBandWidth(device_id);

    if(((size * sizeof(float)) / 1e9) > maxMemory)
        return_value = -2; // Return Value for Hitting OOM condition
    
    cpu_in = (float*)malloc(size * sizeof(float));
    cpu_out = (float*)malloc(2 * sizeof(float));
    
    fillWithRand(cpu_in, size);

    unsigned long long int delta_accumalated = 0;
    
    for(int i=0; i < NUM_ITER; i++)
        delta_accumalated += getMinMax(cpu_in, d_in, d_out, size, cpu_out);

    float average_time = (float)delta_accumalated / NUM_ITER;
    float throughput   = size * sizeof(float) * 0.001f / average_time;

    float max_cpu = cpu_max(cpu_in, size);
    float min_cpu = cpu_min(cpu_in, size);

    if(max_cpu == cpu_out[1] && min_cpu == cpu_out[0])
        return_value = 0;
    else
        return_value = 1;
    
    std::string s;

    if (return_value == 0)
        s = "Pass";
    else if(return_value == 1)
        s = "Fail";
    else if(return_value == -2)
        s = "OOM";

    printf(" %7.0d \t %0.2f \t\t %0.2f % \t %0.1f \t\t %s \n", size, throughput,
		(throughput/bandwidth)*100.0f,average_time, s);
}


void runMinMaxTests(void)
{
    printf("\n N \t\t [GB/s] \t\t [BW Usage] \t [usec] \t Test \n");
    // Big Good Numbers
    int base = 1024;

    for(int i=1; i < 8; i++)
        testMinMax((size_t)std::pow(base, i), 0);

    // Big Bad Numbers

    for(int i=i; i < 5; i++)
        testMinMax((size_t)(std::pow(base, i) + 13533 + i), 0);
}

#endif