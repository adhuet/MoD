#include <cuda_runtime.h>

#include "mod_GPU.hpp"

// https://forums.developer.nvidia.com/t/atomicmin-on-char-is-there-a-way-to-compare-char-to-in-to-use-atomicmin/22246/2
__device__ uchar atomicMinUChar(uchar *address, uchar val)
{
    unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
    unsigned int selectors[] = { 0x3214, 0x3240, 0x3410, 0x4210 };
    unsigned int sel = selectors[(size_t)address & 3];
    unsigned int old, assumed, min_, new_;

    old = *base_address;
    do
    {
        assumed = old;
        min_ =
            min(val, (char)__byte_perm(old, 0, ((size_t)address & 3) | 0x4440));

        new_ = __byte_perm(old, min_, sel);

        if (new_ == old)
            break;
        old = atomicCAS(base_address, assumed, new_);

    } while (assumed != old);

    return old;
}

__device__ int find(int *L, int a)
{
    while (L[a] != a)
        a = L[a];
    return a;
}

__device__ void compress(int *L, int a)
{
    L[a] = find(L, a);
}

__device__ int inlineCompress(int *L, int a)
{
    int id = a;
    while (L[a] != a)
    {
        a = L[a];
        L[id] = a;
    }
    return a;
}

__device__ void merge(int *L, int a, int b)
{
    bool done = false;
    uchar old;
    while (!done)
    {
        a = find(L, a);
        b = find(L, b);
        if (a < b)
        {
            old = atomicMin(&L[b], a);
            done = (old == b);
            b = old;
        }
        else if (b < a)
        {
            old = atomicMin(&L[a], b);
            done = (old == a);
            a = old;
        }
        else
            done = true;
    }
}

__global__ void initCCL(const uchar *src, int *dst, int height, int width)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = y * width + x;

    if (src[idx] != 0)
        dst[idx] = idx;
    else
        dst[idx] = 0;
}

__global__ void mergeCCL(int *dst, int height, int width)
{}

__global__ void compressCCL(int *dst, int height, int width)
{}

__host__ void connectedComponentsGPU(const uchar *src, int *dst, int height,
                                     int width, dim3 gridDim, dim3 blockDim)
{
    initCCL<<<gridDim, blockDim>>>(src, dst, height, width);
}
