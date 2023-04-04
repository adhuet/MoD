#ifndef __MOD_GPU_HPP__
#define __MOD_GPU_HPP__

#include <cstddef>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <vector_types.h>

cv::Mat detectObjectInFrameGPU(const cv::Mat &background, cv::Mat frame);

__global__ void grayscaleGPU(const uchar3 *src, unsigned char *dst, int rows,
                             int cols);

#endif