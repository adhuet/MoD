#include <cuda_runtime.h>

#include "mod_GPU.hpp"

// Get the neighbours of the (x, y) pixel with respect to the Rosenfeld mask:
// p q r
// s x
// __device__ std::vector<int> rosenfeldNeighbours(int *L, int width, int x, int
// y)
// {
//     std::vector<int> neighbours;

//     // Not on top edge
//     if (y > 0)
//     {
//         if (x > 0 && L[(y - 1) * width + (x - 1)] != 0)
//             neighbours.push_back(L[(y - 1) * width + (x - 1)]); // p

//         if (L[(y - 1) * width + x] != 0)
//             neighbours.push_back(L[(y - 1) * width + x]); // q

//         if (x < width - 1 && L[(y - 1) * width + (x + 1)] != 0)
//             neighbours.push_back(L[(y - 1) * width + (x + 1)]);
//     }

//     // Not on left edge
//     if (x > 0 && L[y * width + (x - 1)] != 0)
//         neighbours.push_back(L[y * width + (x - 1)]);

//     return neighbours;
// }

__device__ int find(int *L, int index)
{
    int label = L[index];
    while (label - 1 != index)
    {
        index = label - 1;
        label = L[index];
    }
    return index;
}

// __device__ void compress(int *L, int a)
// {
//     L[a] = find(L, a);
// }

// __device__ int inlineCompress(int *L, int a)
// {
//     int id = a;
//     while (L[a] != a)
//     {
//         a = L[a];
//         L[id] = a;
//     }
//     return a;
// }

__device__ void swap(int *a, int *b)
{
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

__device__ void merge(int *L, int a, int b)
{
    bool done = false;
    int old;
    while (!done)
    {
        a = find(L, a);
        b = find(L, b);
        done = (a == b);
        if (a == b)
            done = true;
        else
        {
            if (!done && a > b)
                swap(&a, &b);
            old = atomicMin(&L[b], a + 1);
            done = (old == b + 1);
            b = old - 1;
        }
    }
}

__global__ void initCCL(const uchar *src, int *dst, int height, int width)
{
    // UF algo
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = y * width + x;

    if (src[idx] != 0)
        dst[idx] = idx + 1;
    else
        dst[idx] = 0;
}

__global__ void mergeCCL(const uchar *src, int *dst, int height, int width)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = y * width + x;

    if (src[idx] != 0)
    {
        // Not on top edge
        if (y > 0)
        {
            // top left neighbour
            if (x > 0 && src[(y - 1) * width + (x - 1)] != 0
                && (y - 1) * width + (x - 1) < idx)
                merge(dst, idx, (y - 1) * width + (x - 1));

            // top neighbour
            if (src[(y - 1) * width + x] != 0 && (y - 1) * width + x < idx)
                merge(dst, idx, (y - 1) * width + x);

            // top right neighbour
            if (x < width - 1 && src[(y - 1) * width + (x + 1)] != 0
                && (y - 1) * width + (x + 1) < idx)
                merge(dst, idx, (y - 1) * width + (x + 1));
        }

        // Not on left edge, left neighbour
        if (x > 0 && src[y * width + (x - 1)] != 0 && y * width + (x - 1) < idx)
            merge(dst, idx, y * width + (x - 1));
    }
}

__global__ void compressCCL(const uchar *src, int *dst, int height, int width)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (src[idx] != 0)
        dst[idx] = find(dst, idx) + 1;
}

__host__ void connectedComponentsGPU(const uchar *src, int *dst, int height,
                                     int width, dim3 gridDim, dim3 blockDim)
{
    initCCL<<<gridDim, blockDim>>>(src, dst, height, width);
    mergeCCL<<<gridDim, blockDim>>>(src, dst, height, width);
    compressCCL<<<gridDim, blockDim>>>(src, dst, height, width);
}
