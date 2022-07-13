#include "common.cuh"

float guassKernel[9] = {1.0/16, 2.0/16, 1.0/16, 2.0/16, 4.0/16, 2.0/16, 1.0/16, 2.0/16, 1.0/16};
cudaMemcpyToSymbol(guassianKernel, &guassKernel[0], 9 * sizeof(float));


void goodFeaturesToTrack(float* image, float* R, float* R_copy, float* dx, float* dy, float* mask, int num_rows, 
                         int num_cols, float lambda, std::vector<cv::Point2i>* points)
{

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(num_rows / threads.y, num_cols / threads.x);

    dim3 threads_reduction(PARALLEL_REDUCTION_THREADS);
    dim3 blocks_reduction(((num_rows * num_cols) /  PARALLEL_REDUCTION_THREADS) + 1);

    float* minMax = (float*)malloc(2 * sizeof(float));

    float max_val = -100;
    sobel<<<blocks, threads>>>(image, dx, dy, num_cols);
    cudaDeviceSynchronize();
    getScores<<<blocks, threads>>>(dx, dy, R, num_cols);
    cudaDeviceSynchronize();
    cudaMemcpy(R_copy, R, num_rows * num_cols * sizeof(float), cudaMemcpyDeviceToDevice);
    compute_reduction(R, image, num_rows * num_cols);
    cudaMemcpy(minMax , image, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Max Val = %f\n", minMax[1]);
    filterScores<<<blocks, threads>>>(R_copy, lambda * minMax[1], num_cols);
    cudaDeviceSynchronize();
    cudaMemcpy(mask, R_copy, num_rows * num_cols * sizeof(float), cudaMemcpyDeviceToHost);

    points->clear();
    for(int i=0; i < num_rows; i++)
        for(int j=0; j < num_cols; j++)
            if(mask[i*num_cols + j])
                points->push_back(cv::Point2i(j, i)); 

    printf("Num Points from Custom Implementation = %d\n", points->size());

}
 

uint32_t runGoodFeaturesToTrackBenchMark(std::string image_path, int size)
{
   unsigned long long int start;
   unsigned long long int delta;

   cv::Mat gray, gray_float, grad_x, grad_y;
   cv::Mat image = cv::imread(image_path);
   cv::cvtColor(image, gray, CV_BGR2GRAY);
   gray.convertTo(gray_float, CV_32FC3, 1/255.);
   cv::resize(gray_float, gray_float, cv::Size(size, size));

   printf("Image is : Rows = %d, Cols = %d\n", gray_float.rows, gray_float.cols);
   float* image_gpu, *dx, *dy, *R, *R_copy;
   float *h_dx, *h_dy;

   cudaMalloc(&image_gpu , gray_float.rows * gray_float.cols * sizeof(float));
   cudaMalloc(&dx , gray_float.rows * gray_float.cols * sizeof(float) );
   cudaMalloc(&dy ,  gray_float.rows * gray_float.cols * sizeof(float));
   cudaMalloc(&R  ,  gray_float.rows * gray_float.cols * sizeof(float));
   cudaMalloc(&R_copy ,  gray_float.rows * gray_float.cols * sizeof(float));

   float* image_cpu = (float*)malloc(gray_float.rows * gray_float.cols * sizeof(float));
   float* mask = (float*)malloc(gray_float.rows * gray_float.cols * sizeof(float));
   h_dx = (float*)malloc(gray_float.rows * gray_float.cols * sizeof(float));
   h_dy = (float*)malloc(gray_float.rows * gray_float.cols * sizeof(float));

   std::memcpy(image_cpu, gray_float.ptr<float>(0),  gray_float.rows * gray_float.cols * sizeof(float));
   cudaMemcpy(image_gpu , image_cpu,  gray_float.rows * gray_float.cols * sizeof(float), cudaMemcpyHostToDevice);

   
   std::vector<cv::Point2i> points;
   start = get_clock();

   goodFeaturesToTrack(image_gpu, R, R_copy, dx, dy, mask, gray_float.rows, gray_float.cols, 0.1, &points);
   cudaDeviceSynchronize();

   delta = get_clock() - start;

   cudaMemcpy(h_dx , dx, gray_float.rows * gray_float.cols * sizeof(float), cudaMemcpyDeviceToHost);
   cudaMemcpy(h_dy , dy, gray_float.rows * gray_float.cols * sizeof(float), cudaMemcpyDeviceToHost);

   cv::Mat dx_mat = cv::Mat(gray_float.rows, gray_float.cols, CV_32FC1, h_dx);
   cv::Mat dy_mat = cv::Mat(gray_float.rows, gray_float.cols, CV_32FC1, h_dy);

   return delta;
}


void visGoodFeaturesToTrack(std::string image_path)
{

   cv::Mat gray, gray_float, grad_x, grad_y;
   cv::Mat image = cv::imread(image_path);
   cv::cvtColor(image, gray, CV_BGR2GRAY);
   gray.convertTo(gray_float, CV_32FC3, 1/255.);
   cv::resize(gray_float, gray_float, cv::Size(640, 640));

   printf("Image is : Rows = %d, Cols = %d\n", gray_float.rows, gray_float.cols);
   float* image_gpu, *dx, *dy, *R, *R_copy;
   float *h_dx, *h_dy;

   cudaMalloc(&image_gpu , gray_float.rows * gray_float.cols * sizeof(float));
   cudaMalloc(&dx , gray_float.rows * gray_float.cols * sizeof(float) );
   cudaMalloc(&dy ,  gray_float.rows * gray_float.cols * sizeof(float));
   cudaMalloc(&R  ,  gray_float.rows * gray_float.cols * sizeof(float));
   cudaMalloc(&R_copy ,  gray_float.rows * gray_float.cols * sizeof(float));

   float* image_cpu = (float*)malloc(gray_float.rows * gray_float.cols * sizeof(float));
   float* mask = (float*)malloc(gray_float.rows * gray_float.cols * sizeof(float));
   h_dx = (float*)malloc(gray_float.rows * gray_float.cols * sizeof(float));
   h_dy = (float*)malloc(gray_float.rows * gray_float.cols * sizeof(float));

   std::memcpy(image_cpu, gray_float.ptr<float>(0),  gray_float.rows * gray_float.cols * sizeof(float));
   cudaMemcpy(image_gpu , image_cpu,  gray_float.rows * gray_float.cols * sizeof(float), cudaMemcpyHostToDevice);

   
   std::vector<cv::Point2i> points;

   goodFeaturesToTrack(image_gpu, R, R_copy, dx, dy, mask, gray_float.rows, gray_float.cols, 0.1, &points);
   cudaDeviceSynchronize();

   cudaMemcpy(h_dx , dx, gray_float.rows * gray_float.cols * sizeof(float), cudaMemcpyDeviceToHost);
   cudaMemcpy(h_dy , dy, gray_float.rows * gray_float.cols * sizeof(float), cudaMemcpyDeviceToHost);

   cv::Mat dx_mat = cv::Mat(gray_float.rows, gray_float.cols, CV_32FC1, h_dx);
   cv::Mat dy_mat = cv::Mat(gray_float.rows, gray_float.cols, CV_32FC1, h_dy);

    cv::resize(image, image, cv::Size(640, 640));
    for(auto &p : points)
    {
        cv::circle(image, p, 1, cv::Scalar(0, 0, 255));
    }
    cv::imshow("dx", dx_mat);
    cv::imshow("dy", dy_mat);
    cv::imshow("image", image);
    cv::waitKey(0);
}