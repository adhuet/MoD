#include "mod_GPU.hpp"

// Input method
__global__ void blurTiledGPU(const uchar *src, uchar *dst, int height,
                             int width, float *kernel, size_t ksize)
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
        float sum = 0.0f;
        for (int y = 0; y < ksize; y++)
            for (int x = 0; x < ksize; x++)
                sum += kernel[y * ksize + x]
                    * tile_src[(y + ty) * block_width + (x + tx)];

        // Final boundary check for write
        if (row_o < height && col_o < width)
            dst[row_o * width + col_o] = static_cast<uchar>(round(sum));
    }
}

// FIXME Output method
/*
__global__ void blurTiledGPU(const uchar *src, uchar *dst, int height,
                             int width, float *kernel, size_t ksize)
{
    extern __shared__ uchar tile[];
    size_t tile_width = blockDim.x + 2 * (ksize / 2);

    int radius = ksize / 2;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    // Load left column
    int col_left = (blockIdx.x - 1) * blockDim.x + tx;
    if (tx >= blockDim.x - radius)
        tile[(radius + ty) * tile_width + tx - (blockDim.x - radius)] =
            (col_left < 0) ? 0 : src[row * width + col_left]; // Border check

    // Load right column
    int col_right = (blockIdx.x + 1) * blockDim.x + tx;
    if (tx < radius)
        tile[(radius + ty) * tile_width + radius + blockDim.x + tx] =
            (col_right >= width) ? 0 : src[row * width + col_right];

    // Load top row
    int row_top = (blockIdx.y - 1) * blockDim.y + ty;
    if (ty >= blockDim.y - radius)
        tile[(ty - (blockDim.y - radius)) * tile_width + radius + tx] =
            (row_top < 0) ? 0 : src[row_top * width + col];

    // Load bottom_row
    int row_bottom = (blockIdx.y + 1) * blockDim.y + ty;
    if (ty < radius)
        tile[(radius + blockDim.y + ty) * tile_width + radius + tx] =
            (row_bottom >= height) ? 0 : src[row_bottom * width + col];

    // Load corners
    // Left side
    if (tx == 0)
    {
        // Top-left corner loads bottom-right
        if (ty == 0)
            tile[(tile_width - 1) * tile_width + tile_width - 1] =
                (row_bottom >= height || col_right >= width)
                ? 0
                : src[row_bottom * width + col_right];
        // Bottom-left loads top-right element
        else if (ty == blockDim.y - 1)
            tile[(0) * tile_width + tile_width - 1] =
                (row_top < 0 || col_right >= width)
                ? 0
                : src[row_top * width + col_right];
    }
    // Right side
    if (tx == blockDim.x - 1)
    {
        // Top-right loads bottom-left
        if (ty == 0)
            tile[(tile_width - 1) * tile_width + 0] =
                (row_bottom >= height || col_left < 0)
                ? 0
                : src[row_bottom * width + col_left];
        // Bottom-right loads top-left
        else if (ty == blockDim.y - 1)
            tile[(0) * tile_width + 0] = (row_top < 0 || col_left < 0)
                ? 0
                : src[row_top * width + col_left];
    }
    // Load internal tile elements
    tile[(radius + ty) * tile_width + radius + tx] =
        (row >= height || col >= width) ? 0 : src[row * width + col];
    __syncthreads();

    float sum = 0;
    for (int i = 0; i < ksize; i++) // y
    {
        for (int j = 0; j < ksize; j++) // x
            sum += kernel[i * ksize + j] * tile[(ty + i) * tile_width + tx + j];
    }

    dst[row * width + col] = static_cast<uchar>(round(sum));
}
*/

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

    dst[y * width + x] = static_cast<uchar>(round(sum));
}