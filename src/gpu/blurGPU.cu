#include "mod_GPU.hpp"

__global__ void blurGPU(const uchar *src, uchar *dst, int height, int width,
                        float *kernel, size_t ksize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }

    int radius = ksize / 2;
    float sum = 0;

    for (int i = -radius; i <= radius; i++)
    {
        for (int j = -radius; j <= radius; j++)
        {
            // Skip when out of bounds
            if (y + i >= 0 && y + i < height && x + j >= 0 && x + j < width)
            {
                sum += kernel[(i + radius) * ksize + (j + radius)]
                    * src[(y + i) * width + (x + j)];
            }
        }
    }

    dst[y * width + x] = static_cast<uchar>(sum);
}