#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <iostream>
#include <opencv2/opencv.hpp>

class SImage
{
public:
    int width;
    int height;
    uchar *data;
    size_t size;
    size_t nb_bytes;

    SImage(int width, int height);
    SImage(int width, int height, uchar *buffer);
    SImage(const cv::Mat &mat);
    ~SImage();

    // Copy constructor
    SImage(const SImage &other);
    friend std::ostream &operator<<(std::ostream &os, const SImage &img);

    cv::Mat toCVMat();
};

void displayVideo(cv::VideoCapture video);
void filter2D(const SImage &src, SImage &dst, float *kernel, size_t ksize);
float *getGaussianMatrix(size_t ksize, double sigma);
uchar *getCircleKernel(size_t ksize);

#endif