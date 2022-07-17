#include "util.cuh"

void findBlockSize(int* whichSize, int* num_el){
	const float pretty_big_number = 24.0f*1024.0f*1024.0f;

	float ratio = float((*num_el))/pretty_big_number;


	if(ratio > 0.8f)
		(*whichSize) =  5;
	else if(ratio > 0.6f)
		(*whichSize) =  4;
	else if(ratio > 0.4f)
		(*whichSize) =  3;
	else if(ratio > 0.2f)
		(*whichSize) =  2;
	else
		(*whichSize) =  1;

}

__device__  float getMedian(float* array)
{
    float tmp;
    for(int i=0; i < 9; i++)
    {
#pragma unroll
        for(int j=0; j < 9; j++)
        {  
            if(array[i] > array[j])
            {
                tmp = array[i];
                array[i] = array[j];
                array[j] = tmp;
            }
        }
    }

    return tmp;
}
