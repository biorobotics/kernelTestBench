#include "ringDetector.cuh"

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

__global__ void rgb2HSV(uint8_t* r, uint8_t* g, uint8_t* b, uint8_t* h, uint8_t* s, uint8_t* v, int num_rows, int num_cols)
{
    __shared__ uint8_t smem_r[TILE_SIZE][TILE_SIZE + 1];
    __shared__ uint8_t smem_g[TILE_SIZE][TILE_SIZE + 1];
    __shared__ uint8_t smem_b[TILE_SIZE][TILE_SIZE + 1];

    int32_t x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int32_t y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int32_t idx = y_idx * num_cols + x_idx;

    smem_r[threadIdx.y][threadIdx.x] = r[idx];
    smem_g[threadIdx.y][threadIdx.x] = g[idx];
    smem_b[threadIdx.y][threadIdx.x] = b[idx];

    __syncthreads();

    int32_t local_row = blockIdx.y * TILE_SIZE + threadIdx.x;
    int32_t local_col = blockIdx.x * TILE_SIZE + threadIdx.y;

    uint8_t r_value = smem_r[threadIdx.x][threadIdx.y];
    uint8_t g_value = smem_g[threadIdx.x][threadIdx.y];
    uint8_t b_value = smem_b[threadIdx.x][threadIdx.y];

    float _r = (float)r_value * 0.003921569F;
    float _g = (float)g_value * 0.003921569F;
    float _b = (float)b_value * 0.003921569F;

    float max, min;
    max = fmaxf(_r, _g);
    max = fmaxf(max, _b);

    min = fminf(_r, _g);
    min = fminf(min, _b);

    float delta = max - min;
    float hue, saturation, value;
    value = max;

    if(max == 0)
    {
        saturation = 0.0f;
        hue = 0.0f;
    }

    else
    {
        saturation = max / delta;
    }

    float nCr = (max - _r) / delta;
    float nCg = (max - _g) / delta;
    float nCb = (max - _b) / delta;

    if(_r == max)
    {
        hue = nCb - nCg;
    }
    else if(_g == max)
    {
        hue = 2.0f + nCr - nCb;
    }
    else if(_b == max)
    {
        hue = 4.0f + nCg - nCr;
    }

    hue = hue * 0.166667F;
    if(hue < 0.0f)
        hue = hue + 1.0f;

    uint8_t hue_8u = (uint8_t)(hue * 255.0f);
    uint8_t saturation_8u = (uint8_t)(saturation * 255.0f);
    uint8_t value_8u = (uint8_t)(value * 255.0f);

    h[local_row * num_cols + local_col] = hue_8u;
    s[local_row * num_cols + local_col] = saturation_8u;
    v[local_row * num_cols + local_col] = value_8u;

}



__global__ void createHSVMask(uint8_t* h, uint8_t* s, uint8_t* v, int num_cols, int num_rows, uint8_t* hsv_mask, 
                        uint8_t _h_ml1, uint8_t _s_ml1, uint8_t _v_ml1, uint8_t _h_mh1, uint8_t _s_mh1, uint8_t _v_mh1, 
                        uint8_t _h_ml2, uint8_t _s_ml2, uint8_t _v_ml2, uint8_t _h_mh2, uint8_t _s_mh2, uint8_t _v_mh2)
{
    int loop_iter = ((TILE_SIZE + 2) * (TILE_SIZE + 2)) / ((TILE_SIZE) * (TILE_SIZE));

    __shared__ float smem_h[TILE_SIZE + 2][TILE_SIZE + 2];
    __shared__ float smem_s[TILE_SIZE + 2][TILE_SIZE + 2];
    __shared__ float smem_v[TILE_SIZE + 2][TILE_SIZE + 2];

    int out = threadIdx.y * TILE_SIZE + threadIdx.x;
    int outY = out / (TILE_SIZE + 2); 
    int outX = out % (TILE_SIZE + 2);

    int local_x = blockIdx.x * TILE_SIZE + outX - 1;
    int local_y = blockIdx.y * TILE_SIZE + outY - 1;

    int local_idx = local_y * num_cols + local_x;

    smem_h[outY][outX] = h[local_idx];
    smem_s[outY][outX] = s[local_idx];
    smem_v[outY][outX] = v[local_idx];

    for(int iter = 1; iter <= loop_iter; iter++)
    {
        out = threadIdx.y  * TILE_SIZE + threadIdx.x + TILE_SIZE * TILE_SIZE;
        outY = out / (TILE_SIZE + 2);
        outX = out % (TILE_SIZE + 2);

        local_x = blockIdx.x * TILE_SIZE + outX - 1;
        local_y = blockIdx.y * TILE_SIZE + outY - 1;
        local_idx = local_y * num_cols + local_x;

        smem_h[outY][outX] = h[local_idx];
        smem_s[outY][outX] = s[local_idx];
        smem_v[outY][outX] = v[local_idx];

    }

    __syncthreads();

    float _h_3x3[9];
    float _s_3x3[9];
    float _v_3x3[9];

    for(int i=0; i < 3; i++)
        for(int j=0; j < 3; j++)
        {
            _h_3x3[i * 3 + j] = smem_h[threadIdx.y + i][threadIdx.x + j];
            _s_3x3[i * 3 + j] = smem_s[threadIdx.y + i][threadIdx.x + j];
            _v_3x3[i * 3 + j] = smem_v[threadIdx.y + i][threadIdx.x + j];
        }

    float _h_median = getMedian(_h_3x3);
    float _s_median = getMedian(_s_3x3);
    float _v_median = getMedian(_v_3x3);

    uint8_t mask1 = (_h_ml1 <= _h_median <= _h_mh1) && (_s_ml1 <= _s_median <= _s_mh1) && (_v_ml1 <= _v_median <= _v_mh1);
    uint8_t mask2 = (_h_ml2 <= _h_median <= _h_mh2) && (_s_ml2 <= _s_median <= _s_mh2) && (_v_ml2 <= _v_median <= _v_mh2);

    uint8_t mask = mask1 | mask2;

    int32_t y = blockIdx.y * TILE_SIZE + threadIdx.y;
    int32_t x = blockIdx.x * TILE_SIZE + threadIdx.x;

    hsv_mask[y * num_cols + x] = mask;

}



__global__ void _fisheye_bitwise_and(uint8_t* hsv_mask, uint8_t* fisheye_mask, int num_cols)
{
    __shared__ uint8_t hsv_shared[TILE_SIZE][TILE_SIZE + 1];
    __shared__ uint8_t fisheye_shared[TILE_SIZE][TILE_SIZE + 1];
    
    int32_t row_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int32_t col_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int32_t idx     = row_idx * num_cols + col_idx;
    
    hsv_shared[threadIdx.y][threadIdx.x] = hsv_mask[idx];
    fisheye_shared[threadIdx.y][threadIdx.x] = fisheye_mask[idx];

    __syncthreads();

    int32_t local_row = blockIdx.y * TILE_SIZE + threadIdx.x;
    int32_t local_col = blockIdx.x * TILE_SIZE + threadIdx.y;

    hsv_mask[local_row * num_cols + local_col] =  hsv_shared[threadIdx.x][threadIdx.y] & fisheye_shared[threadIdx.x][threadIdx.y];

}


