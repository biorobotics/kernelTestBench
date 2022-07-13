#include "common.h"


void visualizeShiTomasiResult(std::string image_path)
{
    visGoodFeaturesToTrack(image_path);
}


void benchmarkShiTomasi(std::string image_path, int size)
{
    int num_iter = 250;
    unsigned long long int time = 0;

    int block_size = size / 32;

    float bandwidth = getPeakBandWidth(0);

    for(int i=0; i < num_iter; i++)
        time += runGoodFeaturesToTrackBenchMark(image_path, size);

    float average_time = time / num_iter;

    float GFLOPS = (126 * size * size * 0.001f) / average_time;
    float BW     = (((((32 + 2) * (32 + 2)) * block_size ) + (2 * size * size) + (((32 + 2) * (32 + 2)) * block_size * 2) + (3 * size * size) ) * 0.001f ) / average_time;

    printf("\t%d   \t%f   \t%f   \t%f  \t%f \n\n", size, GFLOPS, BW, BW / bandwidth, average_time); // Image size, GFLOPS, BW, BW %
}



void runShiTomasiTest()
{
    visualizeShiTomasiResult("checkerboard.png");
    visualizeShiTomasiResult("tiles.jpg");
    visualizeShiTomasiResult("grayscale.jpg");


    printf("\t%s   \t%s   \t%s   \t%s  \t%s \n\n", "NxN Size", "GFLOPS", "BW", "BW Achieved",  "TIME in us");

    benchmarkShiTomasi("grayscale.jpg", 640);
    benchmarkShiTomasi("grayscale.jpg", 1024);
    benchmarkShiTomasi("grayscale.jpg", 2048);
    benchmarkShiTomasi("grayscale.jpg", 4096);
    benchmarkShiTomasi("grayscale.jpg", 8192);

}