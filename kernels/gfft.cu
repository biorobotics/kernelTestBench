#include "gfft.cuh"

__global__ void sobel(float* x, float* dx, float* dy, int num_cols)
{
    __shared__ float smem[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    int32_t tx = threadIdx.x;
    int32_t ty = threadIdx.y;

    int32_t col_start = blockIdx.x * blockDim.x;
    int32_t row_start = blockIdx.y * blockDim.y;

    int32_t col_prev  = (blockIdx.x == 0 ? blockIdx.x : blockIdx.x - 1) * blockDim.x;
    int32_t row_prev  = (blockIdx.y == 0 ? blockIdx.y : blockIdx.y - 1) * blockDim.y;

    int32_t col_next = (blockIdx.x == gridDim.x - 1 ? blockIdx.x : blockIdx.x + 1) * blockDim.x;
    int32_t row_next = (blockIdx.y == gridDim.y - 1 ? blockIdx.y : blockIdx.y + 1) * blockDim.y;

    smem[threadIdx.y + 1][threadIdx.x + 1] = x[(row_start + threadIdx.y) * num_cols + (col_start + threadIdx.x)];
    
    smem[threadIdx.y + 1][0]  = x[(row_start + threadIdx.y) * num_cols + (col_prev + BLOCK_SIZE - 1)];
    smem[threadIdx.y + 1][BLOCK_SIZE + 1] = x[(row_start + threadIdx.y) * num_cols + (col_next + 0)];
    smem[0][threadIdx.x + 1] = x[(row_prev + BLOCK_SIZE - 1) * num_cols + (col_start + threadIdx.x)];
    smem[BLOCK_SIZE + 1][threadIdx.x + 1] = x[(row_next + 0) * num_cols + (col_start + threadIdx.x)];

    smem[0][0] = x[(row_prev + BLOCK_SIZE - 1) * num_cols + (col_prev + BLOCK_SIZE - 1)];
    smem[0][BLOCK_SIZE + 1] = x[(row_prev + BLOCK_SIZE - 1) * num_cols + (col_next + 0)];
    smem[BLOCK_SIZE + 1][0] = x[(row_next + 0) * num_cols + (col_prev + BLOCK_SIZE - 1)];
    smem[BLOCK_SIZE + 1][BLOCK_SIZE + 1] = x[(row_next + 0) * num_cols + (col_next + 0)];

    __syncthreads();

    ++tx;
    ++ty;

    float grad_x = 0;
    float grad_y = 0;

    grad_x = smem[ty-1][tx-1] - smem[ty-1][tx+1] + \
             2 * smem[ty][tx - 1]     - 2 * smem[ty][tx + 1]+ \
             1 * smem[ty + 1][tx - 1]  - smem[ty+1][tx + 1];

    grad_y = smem[ty-1][tx-1] + 2 * smem[ty-1][tx] + smem[ty-1][tx+1] + \
             (-1*smem[ty+1][tx-1]) - 2 * smem[ty+1][tx]  - smem[ty+1][tx+1];

    int32_t local_row = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    int32_t local_col = blockIdx.x * BLOCK_SIZE + threadIdx.y;
    
    dx[(row_start + threadIdx.y)*num_cols + (col_start + threadIdx.x)] = grad_x;
    dy[(row_start + threadIdx.y)*num_cols + (col_start + threadIdx.x)] = grad_y;

}


__global__ void getScores(float* dx, float* dy, float* R, int num_cols)
{
    __shared__ float dx_smem[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
    __shared__ float dy_smem[BLOCK_SIZE + 2][BLOCK_SIZE + 2];


    int32_t tx = threadIdx.x;
    int32_t ty = threadIdx.y;

    int32_t col_start = blockIdx.x * blockDim.x;
    int32_t row_start = blockIdx.y * blockDim.y;

    int32_t col_prev  = (blockIdx.x == 0 ? blockIdx.x : blockIdx.x - 1) * blockDim.x;
    int32_t row_prev  = (blockIdx.y == 0 ? blockIdx.y : blockIdx.y - 1) * blockDim.y;

    int32_t col_next = (blockIdx.x == gridDim.x - 1? blockIdx.x : blockIdx.x + 1) * blockDim.x;
    int32_t row_next = (blockIdx.y == gridDim.y - 1 ? blockIdx.y : blockIdx.y + 1) * blockDim.y;

    dx_smem[threadIdx.y + 1][threadIdx.x + 1] = dx[(row_start + threadIdx.y) * num_cols + (col_start + threadIdx.x)];
    
    dx_smem[threadIdx.y + 1][0]  = dx[(row_start + threadIdx.y) * num_cols + (col_prev + BLOCK_SIZE - 1)];
    dx_smem[threadIdx.y + 1][BLOCK_SIZE + 1] = dx[(row_start + threadIdx.y) * num_cols + (col_next + 0)];
    dx_smem[0][threadIdx.x + 1] = dx[(row_prev + BLOCK_SIZE - 1) * num_cols + (col_start + threadIdx.x)];
    dx_smem[BLOCK_SIZE + 1][threadIdx.x + 1] = dx[(row_next + 0) * num_cols + (col_start + threadIdx.x)];

    dx_smem[0][0] = dx[(row_prev + BLOCK_SIZE - 1) * num_cols + (col_prev + BLOCK_SIZE - 1)];
    dx_smem[0][BLOCK_SIZE + 1] = dx[(row_prev + BLOCK_SIZE - 1) * num_cols + (col_next + 0)];
    dx_smem[BLOCK_SIZE + 1][0] = dx[(row_next + 0) * num_cols + (col_prev + BLOCK_SIZE - 1)];
    dx_smem[BLOCK_SIZE + 1][BLOCK_SIZE + 1] = dx[(row_next + 0) * num_cols + (col_next + 0)];

    dy_smem[threadIdx.y + 1][threadIdx.x + 1] = dy[(row_start + threadIdx.y) * num_cols + (col_start + threadIdx.x)];
    
    dy_smem[threadIdx.y + 1][0]  = dy[(row_start + threadIdx.y) * num_cols + (col_prev + BLOCK_SIZE - 1)];
    dy_smem[threadIdx.y + 1][BLOCK_SIZE + 1] = dy[(row_start + threadIdx.y) * num_cols + (col_next + 0)];
    dy_smem[0][threadIdx.x + 1] = dy[(row_prev + BLOCK_SIZE - 1) * num_cols + (col_start + threadIdx.x)];
    dy_smem[BLOCK_SIZE + 1][threadIdx.x + 1] = dy[(row_next + 0) * num_cols + (col_start + threadIdx.x)];

    dy_smem[0][0] = dx[(row_prev + BLOCK_SIZE - 1) * num_cols + (col_prev + BLOCK_SIZE - 1)];
    dy_smem[0][BLOCK_SIZE + 1] = dx[(row_prev + BLOCK_SIZE - 1) * num_cols + (col_next + 0)];
    dy_smem[BLOCK_SIZE + 1][0] = dx[(row_next + 0) * num_cols + (col_prev + BLOCK_SIZE - 1)];
    dy_smem[BLOCK_SIZE + 1][BLOCK_SIZE + 1] = dx[(row_next + 0) * num_cols + (col_next + 0)];

    __syncthreads();
    
    ++tx;
    ++ty;

    int w_offset_row = 1;
    int w_offset_col = 1;

    float dxx = 0;
    float dyy = 0;
    float dxy = 0;

    for(int i=-1; i <= 1; i++)
#pragma unroll
        for(int j=-1; j <= 1; j++)
        {
            //dxx += guassianKernel[(i+w_offset_row) * 3 + (j+w_offset_col)] * dx_smem[ty + i][tx + j] * dx_smem[ty + i][tx + j];
            //dyy += guassianKernel[(i+w_offset_row) * 3 + (j+w_offset_col)] * dy_smem[ty + i][tx + j] * dy_smem[ty + i][tx + j];
            //dxy += guassianKernel[(i+w_offset_row) * 3 + (j+w_offset_col)] * dx_smem[ty + i][tx + j] * dy_smem[ty + i][tx + j];

            dxx += dx_smem[ty + i][tx + j] * dx_smem[ty + i][tx + j];
            dyy += dy_smem[ty + i][tx + j] * dy_smem[ty + i][tx + j];
            dxy += dx_smem[ty + i][tx + j] * dy_smem[ty + i][tx + j];

        }

    float score = (dxx + dyy + sqrtf(((dxx - dyy) * (dxx - dyy)) + 4 * dxy * dxy)) / 2;

    R[(row_start + threadIdx.y)*num_cols + (col_start + threadIdx.x)] = score;
}


__global__ void filterScores(float* R, float thresh, int num_cols)
{
    __shared__ float  smem[BLOCK_SIZE][BLOCK_SIZE + 1];
    int32_t row_num = blockIdx.y * blockDim.y + threadIdx.y;
    int32_t col_num = blockIdx.x * blockDim.x + threadIdx.x;
    
    smem[threadIdx.y][threadIdx.x] = R[row_num * num_cols + col_num];

    __syncthreads();

    R[row_num * num_cols + col_num] = smem[threadIdx.y][threadIdx.x] > thresh;
}

