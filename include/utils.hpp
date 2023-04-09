#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <opencv2/opencv.hpp>

// typedef struct image s_image;
struct s_image
{
    int width;
    int height;
    uchar *data;
};

// class SImage
// {
// public:
//     int width;
//     int height;
//     uchar *data;

//     SImage(int width, int height);
//     SImage(const cv::Mat &mat);
//     ~SImage();

//     SImage operator=(const SImage &other);
// };

void displayVideo(cv::VideoCapture video);
s_image toPtr(const cv::Mat &image);
cv::Mat toMat(s_image image);
void filter2D(s_image src, s_image dst, float *kernel, size_t ksize);
float *getGaussianMatrix(size_t ksize, double sigma);
uchar *getCircleKernel(size_t ksize);

#endif