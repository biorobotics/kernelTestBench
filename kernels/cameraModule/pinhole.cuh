#include <cuda.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <opencv2/opencv.hpp>


#define FOCAL_LENGHT 40
#define COL 1024
#define ROW 1024
#define NUM_ITER 8



__global__ void liftProjective_0_rejectWithF(cv::Point2f* cur_points, cv::Point2f* fow_points, cv::Point2f* un_cur_points, cv::Point2f* un_fow_points, 
                double m_inv_k11, double m_inv_k13, double m_inv_k22, double m_inv_k23, 
                double k1, double k2, double p1, double p2, size_t num_elements);



__global__ void liftProjective_recursive_rejectWithF(cv::Point2f* cur_points, cv::Point2f* fow_points, cv::Point2f* un_cur_points, cv::Point2f* un_fow_points, 
                double m_inv_k11, double m_inv_k13, double m_inv_k22, double m_inv_k23, 
                double k1, double k2, double p1, double p2, size_t num_elements);


__global__ void liftProjective_noDistort_rejectWithF(cv::Point2f* cur_points, cv::Point2f* fow_points, cv::Point2f* un_cur_points, cv::Point2f* un_fow_points, 
                double m_inv_k11, double m_inv_k13, double m_inv_k22, double m_inv_k23, 
                double k1, double k2, double p1, double p2, size_t num_elements);


