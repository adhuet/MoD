#include "mod_GPU.hpp"

__global__ void thresholdGPU(const uchar *src, uchar *dst, int height,
                             int width, uchar threshold, uchar maxval)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }

    if (src[y * width + x] < threshold)
        dst[y * width + x] = 0;
    else
        dst[y * width + x] = maxval;
}