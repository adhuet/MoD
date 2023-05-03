#include "mod_GPU.hpp"

__global__ void dilateTiledGPU(const uchar *src, uchar *dst, int height,
                               int width, uchar *circleKernel, size_t ksize)
{
    extern __shared__ uchar tile_src[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tile_width = blockDim.x - ksize + 1;
    int block_width = blockDim.x;

    // Get the output indices
    int row_o = ty + blockIdx.y * tile_width;
    int col_o = tx + blockIdx.x * tile_width;

    // Input, tile-loading indices are output plus the kernel radius
    int row_i = row_o - ksize / 2;
    int col_i = col_o - ksize / 2;

    // Load tile elements
    if (row_i >= 0 && row_i < height && col_i >= 0 && col_i < width)
        tile_src[ty * block_width + tx] = src[row_i * width + col_i];
    else
        tile_src[ty * block_width + tx] = 0.0f;

    // Wait until all tile elements are loaded
    __syncthreads();

    // Only compute if thread is a writer (i.e. within tile)
    if (tx < tile_width && ty < tile_width)
    {
        uchar value = 0;
        for (int y = 0; y < ksize; y++)
        {
            for (int x = 0; x < ksize; x++)
            {
                uchar kernelValue = circleKernel[y * ksize + x];
                // Skip 0 elements
                if (!kernelValue)
                    continue;

                uchar pixel = tile_src[(y + ty) * block_width + (x + tx)];
                value |= pixel & kernelValue;
            }
        }

        // Final boundary check for write
        if (row_o < height && col_o < width)
            dst[row_o * width + col_o] = value == 1 ? 255 : 0;
    }
}

__global__ void erodeTiledGPU(const uchar *src, uchar *dst, int height,
                              int width, uchar *circleKernel, size_t ksize)
{
    extern __shared__ uchar tile_src[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tile_width = blockDim.x - ksize + 1;
    int block_width = blockDim.x;

    // Get the output indices
    int row_o = ty + blockIdx.y * tile_width;
    int col_o = tx + blockIdx.x * tile_width;

    // Input, tile-loading indices are output plus the kernel radius
    int row_i = row_o - ksize / 2;
    int col_i = col_o - ksize / 2;

    // Load tile elements
    if (row_i >= 0 && row_i < height && col_i >= 0 && col_i < width)
        tile_src[ty * block_width + tx] = src[row_i * width + col_i];
    else
        tile_src[ty * block_width + tx] = 0.0f;

    // Wait until all tile elements are loaded
    __syncthreads();

    // Only compute if thread is a writer (i.e. within tile)
    if (tx < tile_width && ty < tile_width)
    {
        uchar value = 1;
        for (int y = 0; y < ksize; y++)
        {
            for (int x = 0; x < ksize; x++)
            {
                uchar kernelValue = circleKernel[y * ksize + x];
                // Skip 0 elements
                if (!kernelValue)
                    continue;

                uchar pixel = tile_src[(y + ty) * block_width + (x + tx)];
                value &= pixel & kernelValue;
            }
        }

        // Final boundary check for write
        if (row_o < height && col_o < width)
            dst[row_o * width + col_o] = value == 1 ? 255 : 0;
    }
}

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
