
#include "laserExtractor.cuh"


void extractLaserRing(uint8_t *r, uint8_t* g, uint8_t* b, uint8_t _h_ml1, uint8_t _s_ml1, uint8_t _v_ml1, uint8_t _h_mh1, uint8_t _s_mh1, uint8_t _v_mh1, 
                      uint8_t _h_ml2, uint8_t _s_ml2, uint8_t _v_ml2, uint8_t _h_mh2, uint8_t _s_mh2, uint8_t _v_mh2, 
                      uint8_t* fisheye_mask, int num_rows, int num_cols, uint8_t* h, uint8_t* s, uint8_t* v, uint8_t* dilate_kernel, uint8_t* close_kernel, uint8_t* hsv_mask,
                      uint8_t* intermediateResult)
{
    NppiSize dilate_kernel_size{3, 3};
    NppiSize close_kernel_size{5, 5};
    NppiPoint anchor = {-1, -1};

    NppiSize sizeROI = {num_cols, num_rows};

    dim3 threads(32, 32);
    dim3 blocks(num_rows / threads.x, num_cols / threads.y);

    rgb2HSV<<<blocks, threads>>>(r, g, b, h, s, v, num_rows, num_cols);
    cudaDeviceSynchronize();
    createHSVMask<<<blocks, threads>>>(h, s, v, num_cols, num_rows, hsv_mask, 
                        _h_ml1, _s_ml1,  _v_ml1,  _h_mh1,  _s_mh1,  _v_mh1, 
                        _h_ml2,  _s_ml2,  _v_ml2,  _h_mh2,  _s_mh2,  _v_mh2);
    cudaDeviceSynchronize();

   nppiDilate_8u_C1R(hsv_mask, num_cols, intermediateResult, num_cols, sizeROI, dilate_kernel, dilate_kernel_size, anchor);
   cudaDeviceSynchronize();

   nppiDilate_8u_C1R(intermediateResult, num_cols, hsv_mask, num_cols, sizeROI, close_kernel, close_kernel_size, anchor);
   cudaDeviceSynchronize();

   nppiErode_8u_C1R(hsv_mask, num_cols, intermediateResult, num_cols, sizeROI, close_kernel, close_kernel_size, anchor);
   cudaDeviceSynchronize();

   // Min max Here;

}
