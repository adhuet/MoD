#include "mod.hpp"

void dilateBinary255(const SImage &src, SImage &dst, uchar *kernel,
                     size_t ksize)
{
    for (int y = 0; y < src.height; y++)
    {
        for (int x = 0; x < src.width; x++)
        {
            int radius = ksize / 2;
            uchar value = 0;
            for (int i = -radius; i <= radius; i++)
            {
                for (int j = -radius; j <= radius; j++)
                {
                    uchar kernelValue =
                        kernel[(i + radius) * ksize + (j + radius)];
                    // We can skip 0 elements
                    if (!kernelValue)
                        continue;

                    // 0 if we are out of bounds
                    uchar pixel = 0;
                    // In-bounds, so take the value
                    if (y + i >= 0 && y + i < src.height && x + j >= 0
                        && x + j < src.width)
                        pixel = src.data[(y + i) * src.width + (x + j)];

                    value |= pixel & kernelValue;
                }
            }
            dst.data[y * dst.width + x] = value == 1 ? 255 : 0;
        }
    }
}

void erodeBinary255(const SImage &src, SImage &dst, uchar *kernel, size_t ksize)
{
    for (int y = 0; y < src.height; y++)
    {
        for (int x = 0; x < src.width; x++)
        {
            int radius = ksize / 2;
            uchar value = 1;
            for (int i = -radius; i <= radius; i++)
            {
                for (int j = -radius; j <= radius; j++)
                {
                    uchar kernelValue =
                        kernel[(i + radius) * ksize + (j + radius)];
                    // We can skip 0 elements
                    if (!kernelValue)
                        continue;

                    // 0 if we are out of bounds
                    uchar pixel = 0;
                    // In-bounds, so take the value
                    if (y + i >= 0 && y + i < src.height && x + j >= 0
                        && x + j < src.width)
                        pixel = src.data[(y + i) * src.width + (x + j)];

                    value &= pixel & kernelValue;
                }
            }
            dst.data[y * dst.width + x] = value == 1 ? 255 : 0;
        }
    }
}

void dilateBinary1(const SImage &src, SImage &dst, uchar *kernel, size_t ksize)
{
    for (int y = 0; y < src.height; y++)
    {
        for (int x = 0; x < src.width; x++)
        {
            int radius = ksize / 2;
            uchar value = 0;
            for (int i = -radius; i <= radius; i++)
            {
                for (int j = -radius; j <= radius; j++)
                {
                    uchar kernelValue =
                        kernel[(i + radius) * ksize + (j + radius)];
                    // We can skip 0 elements
                    if (!kernelValue)
                        continue;

                    // 0 if we are out of bounds
                    uchar pixel = 0;
                    // In-bounds, so take the value
                    if (y + i >= 0 && y + i < src.height && x + j >= 0
                        && x + j < src.width)
                        pixel = src.data[(y + i) * src.width + (x + j)];

                    value |= pixel & kernelValue;
                }
            }
            dst.data[y * dst.width + x] = value;
        }
    }
}

void erodeBinary1(const SImage &src, SImage &dst, uchar *kernel, size_t ksize)
{
    for (int y = 0; y < src.height; y++)
    {
        for (int x = 0; x < src.width; x++)
        {
            int radius = ksize / 2;
            uchar value = 1;
            for (int i = -radius; i <= radius; i++)
            {
                for (int j = -radius; j <= radius; j++)
                {
                    uchar kernelValue =
                        kernel[(i + radius) * ksize + (j + radius)];
                    // We can skip 0 elements
                    if (!kernelValue)
                        continue;

                    // 0 if we are out of bounds
                    uchar pixel = 0;
                    // In-bounds, so take the value
                    if (y + i >= 0 && y + i < src.height && x + j >= 0
                        && x + j < src.width)
                        pixel = src.data[(y + i) * src.width + (x + j)];

                    value &= pixel & kernelValue;
                }
            }
            dst.data[y * dst.width + x] = value;
        }
    }
}

void morphOpen(const SImage &src, SImage &dst, size_t ksize)
{
    uchar *kernel = getCircleKernel(ksize);
    SImage tmp = src;

    dilateBinary255(src, tmp, kernel, ksize);
    erodeBinary255(tmp, dst, kernel, ksize);
    delete[] kernel;
}

void morphClose(const SImage &src, SImage &dst, size_t ksize)
{
    uchar *kernel = getCircleKernel(ksize);
    SImage tmp = src;
    erodeBinary255(src, tmp, kernel, ksize);
    dilateBinary255(tmp, dst, kernel, ksize);
    delete[] kernel;
}
