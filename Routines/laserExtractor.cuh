#ifndef LASEREXTRACTOR_CUH
#define LASEREXTRACTOR_CUH

#include "common.cuh"

void extractLaserRing(uint8_t *r, uint8_t* g, uint8_t* b, uint8_t _h_ml1, uint8_t _s_ml1, uint8_t _v_ml1, uint8_t _h_mh1, uint8_t _s_mh1, uint8_t _v_mh1, 
                      uint8_t _h_ml2, uint8_t _s_ml2, uint8_t _v_ml2, uint8_t _h_mh2, uint8_t _s_mh2, uint8_t _v_mh2, 
                      uint8_t* fisheye_mask, int num_rows, int num_cols, uint8_t* h, uint8_t* s, uint8_t* v, uint8_t* dilate_kernel, uint8_t* close_kernel, uint8_t* hsv_mask,
                      uint8_t* intermediateResult);


#endif