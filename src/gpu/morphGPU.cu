#include "mod_GPU.hpp"

extern __device__ __constant__ size_t c_morph_diameter;
extern __device__ __constant__ uchar c_circleKernel[];

__global__ void dilateTiledConstantGPU2(const uchar *src, uchar *dst,
                                        int height, int width)
{
    extern __shared__ uchar tile_src[];
    size_t shared_width = blockDim.x + c_morph_diameter - 1;
    size_t tile_width = blockDim.x;
    // 1. Phase to Load Data into Shared Memory. Each Thread loads multiple
    // elements indexed by each Batch loading
    // 1.o_idx: RMO ID 2. o_row & o_col: Row and Column of Shared Memory
    // 3. i_row & i_col: Indexes to fetch data from input Image
    // 4. src: RMO index of Input Image

    // First batch loading
    int o_idx = threadIdx.y * tile_width + threadIdx.x,
        o_row = o_idx / shared_width, o_col = o_idx % shared_width,
        i_row = blockIdx.y * tile_width + o_row - (c_morph_diameter / 2),
        i_col = blockIdx.x * tile_width + o_col - (c_morph_diameter / 2),
        i_idx = (i_row * width + i_col);
    if (i_row >= 0 && i_row < height && i_col >= 0 && i_col < width)
        tile_src[o_row * shared_width + o_col] = src[i_idx];
    else
        tile_src[o_row * shared_width + o_col] = 0.0;

    for (int iter = 1;
         iter <= (shared_width * shared_width) / (tile_width * tile_width);
         iter++)
    {
        // Second batch loading
        o_idx = threadIdx.y * tile_width + threadIdx.x
            + iter * (tile_width * tile_width);
        o_row = o_idx / shared_width, o_col = o_idx % shared_width;
        i_row = blockIdx.y * tile_width + o_row - (c_morph_diameter / 2);
        i_col = blockIdx.x * tile_width + o_col - (c_morph_diameter / 2);
        i_idx = (i_row * width + i_col);
        if (o_row < shared_width && o_col < shared_width)
        {
            if (i_row >= 0 && i_row < height && i_col >= 0 && i_col < width)
                tile_src[o_row * shared_width + o_col] = src[i_idx];
            else
                tile_src[o_row * shared_width + o_col] = 0.0;
        }
    }
    __syncthreads();

    uchar value = 0;
    int y, x;
    for (y = 0; y < c_morph_diameter; y++)
    {
        for (x = 0; x < c_morph_diameter; x++)
        {
            uchar kernelValue = c_circleKernel[y * c_morph_diameter + x];
            if (!kernelValue)
                continue;
            uchar pixel =
                tile_src[(threadIdx.y + y) * shared_width + (threadIdx.x + x)];
            value |= pixel & kernelValue;
        }
    }
    y = blockIdx.y * tile_width + threadIdx.y;
    x = blockIdx.x * tile_width + threadIdx.x;
    if (y < height && x < width)
        dst[(y * width + x)] = value == 1 ? 255 : 0;
}

__global__ void erodeTiledConstantGPU2(const uchar *src, uchar *dst, int height,
                                       int width)
{
    extern __shared__ uchar tile_src[];
    size_t shared_width = blockDim.x + c_morph_diameter - 1;
    size_t tile_width = blockDim.x;
    // 1. Phase to Load Data into Shared Memory. Each Thread loads multiple
    // elements indexed by each Batch loading
    // 1.o_idx: RMO ID 2. o_row & o_col: Row and Column of Shared Memory
    // 3. i_row & i_col: Indexes to fetch data from input Image
    // 4. src: RMO index of Input Image

    // First batch loading
    int o_idx = threadIdx.y * tile_width + threadIdx.x,
        o_row = o_idx / shared_width, o_col = o_idx % shared_width,
        i_row = blockIdx.y * tile_width + o_row - (c_morph_diameter / 2),
        i_col = blockIdx.x * tile_width + o_col - (c_morph_diameter / 2),
        i_idx = (i_row * width + i_col);
    if (i_row >= 0 && i_row < height && i_col >= 0 && i_col < width)
        tile_src[o_row * shared_width + o_col] = src[i_idx];
    else
        tile_src[o_row * shared_width + o_col] = 0.0;

    for (int iter = 1;
         iter <= (shared_width * shared_width) / (tile_width * tile_width);
         iter++)
    {
        // Second batch loading
        o_idx = threadIdx.y * tile_width + threadIdx.x
            + iter * (tile_width * tile_width);
        o_row = o_idx / shared_width, o_col = o_idx % shared_width;
        i_row = blockIdx.y * tile_width + o_row - (c_morph_diameter / 2);
        i_col = blockIdx.x * tile_width + o_col - (c_morph_diameter / 2);
        i_idx = (i_row * width + i_col);
        if (o_row < shared_width && o_col < shared_width)
        {
            if (i_row >= 0 && i_row < height && i_col >= 0 && i_col < width)
                tile_src[o_row * shared_width + o_col] = src[i_idx];
            else
                tile_src[o_row * shared_width + o_col] = 0.0;
        }
    }
    __syncthreads();

    uchar value = 1;
    int y, x;
    for (y = 0; y < c_morph_diameter; y++)
    {
        for (x = 0; x < c_morph_diameter; x++)
        {
            uchar kernelValue = c_circleKernel[y * c_morph_diameter + x];
            if (!kernelValue)
                continue;
            uchar pixel =
                tile_src[(threadIdx.y + y) * shared_width + (threadIdx.x + x)];
            value &= pixel & kernelValue;
        }
    }
    y = blockIdx.y * tile_width + threadIdx.y;
    x = blockIdx.x * tile_width + threadIdx.x;
    if (y < height && x < width)
        dst[(y * width + x)] = value == 1 ? 255 : 0;
}

__global__ void dilateTiledConstantGPU(const uchar *src, uchar *dst, int height,
                                       int width)
{
    extern __shared__ uchar tile_src[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tile_width = blockDim.x - c_morph_diameter + 1;
    int block_width = blockDim.x;

    // Get the output indices
    int row_o = ty + blockIdx.y * tile_width;
    int col_o = tx + blockIdx.x * tile_width;

    // Input, tile-loading indices are output plus the kernel radius
    int row_i = row_o - c_morph_diameter / 2;
    int col_i = col_o - c_morph_diameter / 2;

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
        for (int y = 0; y < c_morph_diameter; y++)
        {
            for (int x = 0; x < c_morph_diameter; x++)
            {
                uchar kernelValue = c_circleKernel[y * c_morph_diameter + x];
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

__global__ void erodeTiledConstantGPU(const uchar *src, uchar *dst, int height,
                                      int width)
{
    extern __shared__ uchar tile_src[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tile_width = blockDim.x - c_morph_diameter + 1;
    int block_width = blockDim.x;

    // Get the output indices
    int row_o = ty + blockIdx.y * tile_width;
    int col_o = tx + blockIdx.x * tile_width;

    // Input, tile-loading indices are output plus the kernel radius
    int row_i = row_o - c_morph_diameter / 2;
    int col_i = col_o - c_morph_diameter / 2;

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
        for (int y = 0; y < c_morph_diameter; y++)
        {
            for (int x = 0; x < c_morph_diameter; x++)
            {
                uchar kernelValue = c_circleKernel[y * c_morph_diameter + x];
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
