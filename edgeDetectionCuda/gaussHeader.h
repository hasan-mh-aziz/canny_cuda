#ifndef GAUSS_H    // To make sure you don't declare the function more than once by including the header multiple times.
#define GAUSS_H


#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

cv::Mat getGaussianKernel(int rows, int cols, double sigmax, double sigmay);

#endif
