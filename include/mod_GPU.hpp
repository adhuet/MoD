#ifndef __MOD_GPU_HPP__
#define __MOD_GPU_HPP__

#include <cstddef>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <vector_types.h>

cv::Mat detectObjectInFrameGPU(const cv::Mat &background, cv::Mat frame);

__global__ void grayscaleGPU(const uchar3 *src, uchar *dst, int height,
                             int width);

__global__ void blurGPU(const uchar *src, uchar *dst, int height, int width,
                        float *kernel, size_t ksize);

__global__ void diffGPU(const uchar *src1, const uchar *src2, uchar *dst, int height, int width);

__global__ void thresholdGPU(const uchar *src, uchar *dst, int height, int width, uchar threshold, uchar maxval);
#endif