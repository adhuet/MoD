#include "mod_GPU.hpp"

__global__ void grayscaleGPU(const uchar3 *src, uchar *dst, int height,
                             int width)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= width || row >= height)
    {
        return;
    }

    uchar3 inputPixel = src[row * width + col];
    uchar grayValue = static_cast<uchar>(
        0.299f * inputPixel.x + 0.587f * inputPixel.y + 0.114f * inputPixel.z);
    dst[row * width + col] = grayValue;
}