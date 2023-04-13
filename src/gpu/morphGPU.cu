#include "mod_GPU.hpp"

__global__ void dilateGPU(const uchar *src, uchar *dst, int height, int width,
                          uchar *circleKernel, size_t ksize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }

    int radius = ksize / 2;
    uchar value = 0;
    for (int i = -radius; i <= radius; i++)
    {
        for (int j = -radius; j <= radius; j++)
        {
            uchar kernelValue =
                circleKernel[(i + radius) * ksize + (j + radius)];
            // We can skip 0 elements
            if (!kernelValue)
                continue;

            // 0 if we are out of bounds
            uchar pixel = 0;
            // In-bounds, so take the value
            if (y + i >= 0 && y + i < height && x + j >= 0 && x + j < width)
                pixel = src[(y + i) * width + (x + j)];

            value |= pixel & kernelValue;
        }
    }
    __syncthreads();
    dst[y * width + x] = value == 1 ? 255 : 0;
}

__global__ void erodeGPU(const uchar *src, uchar *dst, int height, int width,
                         uchar *circleKernel, size_t ksize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }
    int radius = ksize / 2;
    uchar value = 1;
    for (int i = -radius; i <= radius; i++)
    {
        for (int j = -radius; j <= radius; j++)
        {
            uchar kernelValue =
                circleKernel[(i + radius) * ksize + (j + radius)];
            // We can skip 0 elements
            if (!kernelValue)
                continue;

            // 0 if we are out of bounds
            uchar pixel = 0;
            // In-bounds, so take the value
            if (y + i >= 0 && y + i < height && x + j >= 0 && x + j < width)
                pixel = src[(y + i) * width + (x + j)];

            value &= pixel & kernelValue;
        }
    }
    __syncthreads();
    dst[y * width + x] = value == 1 ? 255 : 0;
}
