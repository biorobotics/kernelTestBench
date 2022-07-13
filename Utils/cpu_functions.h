#include "common.h"

float cpu_min(float* in, int num_els)
{
	float min;

	for(int i = 0; i < num_els; i++)
		min = in[i] < min ? in[i] : min;

	return min;
}
float cpu_max(float* in, int num_els)
{
	float max;

	for(int i = 0; i < num_els; i++)
		max = in[i] > max ? in[i] : max;

	return max;
}

void fillWithRand(float* in, size_t num_elements)
{
    for(size_t i=0; i < num_elements; i++)
        in[i] = rand();
}


unsigned long long int get_clock()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (unsigned long long int)tv.tv_usec + 1000000*tv.tv_sec;

}
