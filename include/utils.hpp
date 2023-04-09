#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <iostream>
#include <opencv2/opencv.hpp>

// typedef struct image s_image;
// struct s_image
// {
//     int width;
//     int height;
//     uchar *data;
// };

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

// std::ostream &operator<<(std::ostream &os, const SImage &img)
// {
//     for (int i = 0; i < img.height; i++)
//     {
//         for (int j = 0; j < img.width; j++)
//             std::cout << static_cast<unsigned>(img.data[i * img.width + j])
//                       << " ";
//         std::cout << std::endl;
//     }
//     std::cout << std::endl;
// }

void displayVideo(cv::VideoCapture video);
// s_image toPtr(const cv::Mat &image);
// cv::Mat toMat(s_image image);
void filter2D(const SImage &src, SImage &dst, float *kernel, size_t ksize);
float *getGaussianMatrix(size_t ksize, double sigma);
uchar *getCircleKernel(size_t ksize);

#endif