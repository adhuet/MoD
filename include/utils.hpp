#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <opencv2/opencv.hpp>

struct s_image
{
    int width;
    int height;
    uchar *data;
};

// typedef struct image s_image;

void displayVideo(cv::VideoCapture video);
s_image toPtr(const cv::Mat &image);
cv::Mat toMat(s_image image);
void filter2D(s_image src, s_image dst, float *kernel, size_t ksize);
float *getGaussianMatrix(size_t ksize, double sigma);

#endif