#include "mod_GPU.hpp"

__global__ void diffGPU(const uchar *src1, const uchar *src2, uchar *dst,
                        int height, int width)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }

    dst[y * width + x] = abs(src1[y * width + x] - src2[y * width + x]);
}