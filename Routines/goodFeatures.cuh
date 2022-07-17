#ifndef GOODFEATURES_CUH
#define GOODFEATURES_CUH

#include "minMax.cuh"
#include "common.cuh"



void goodFeaturesToTrack(float* image, float* R, float* R_copy, float* dx, float* dy, float* mask, int num_rows, 
                         int num_cols, float lambda, std::vector<cv::Point2i>* points);


uint32_t runGoodFeaturesToTrackBenchMark(std::string image_path, int size);


void visGoodFeaturesToTrack(std::string image_path);

#endif