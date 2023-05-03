#ifndef __MOD_GPU_HPP__
#define __MOD_GPU_HPP__

#include <cstddef>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <vector_types.h>

#define CUDA_WARN(XXX)                                                         \
    do                                                                         \
    {                                                                          \
        if (XXX != cudaSuccess)                                                \
            std::cerr << "CUDA Error: " << cudaGetErrorString(XXX)             \
                      << ", at line " << __LINE__ << std::endl;                \
        cudaDeviceSynchronize();                                               \
    } while (0)

std::vector<cv::Rect> detectObjectInFrameGPU(const cv::Mat &background,
                                             const cv::Mat &frame);
int renderObjectsInCaptureGPU(cv::VideoCapture capture);

__global__ void grayscaleGPU(const uchar3 *src, uchar *dst, int height,
                             int width);

__global__ void blurGPU(const uchar *src, uchar *dst, int height, int width,
                        float *kernel, size_t ksize);
__global__ void blurTiledGPU(const uchar *src, uchar *dst, int height,
                             int width, float *kernel, size_t ksize);
__global__ void blurTiledConstantGPU(const uchar *src, uchar *dst, int height,
                                     int width);
__global__ void blurTiledGPU2(const uchar *src, uchar *dst, int height,
                              int width, float *kernel, int ksize);
__global__ void blurTiledConstantGPU2(const uchar *src, uchar *dst, int height,
                                      int width);

__global__ void diffGPU(const uchar *src1, const uchar *src2, uchar *dst,
                        int height, int width);

__global__ void thresholdGPU(const uchar *src, uchar *dst, int height,
                             int width, uchar threshold, uchar maxval);

__global__ void dilateGPU(const uchar *src, uchar *dst, int height, int width,
                          uchar *circleKernel, size_t ksize);
__global__ void erodeGPU(const uchar *src, uchar *dst, int height, int width,
                         uchar *circleKernel, size_t ksize);
__global__ void dilateTiledGPU(const uchar *src, uchar *dst, int height,
                               int width, uchar *circleKernel, size_t ksize);
__global__ void erodeTiledGPU(const uchar *src, uchar *dst, int height,
                              int width, uchar *circleKernel, size_t ksize);
__global__ void dilateTiledConstantGPU(const uchar *src, uchar *dst, int height,
                                       int width);
__global__ void erodeTiledConstantGPU(const uchar *src, uchar *dst, int height,
                                      int width);
__global__ void dilateTiledConstantGPU2(const uchar *src, uchar *dst,
                                        int height, int width);
__global__ void erodeTiledConstantGPU2(const uchar *src, uchar *dst, int height,
                                       int width);

__global__ void initCCL(const uchar *src, int *dst, int height, int width);
__global__ void mergeCCL(const uchar *src, int *dst, int height, int width);
__global__ void compressCCL(const uchar *src, int *dst, int height, int width);

__host__ void connectedComponentsGPU(const uchar *src, int *dst, int height,
                                     int width, dim3 gridDim, dim3 blockDim);

#endif