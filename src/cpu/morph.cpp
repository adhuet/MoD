#include "mod.hpp"

void dilateBinary255(s_image src, s_image dst, uchar *kernel, size_t ksize)
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

void erodeBinary255(s_image src, s_image dst, uchar *kernel, size_t ksize)
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

void dilateBinary1(s_image src, s_image dst, uchar *kernel, size_t ksize)
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

void erodeBinary1(s_image src, s_image dst, uchar *kernel, size_t ksize)
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

void morphOpen(s_image src, s_image dst, size_t ksize)
{
    uchar *kernel = getCircleKernel(ksize);
    s_image tmp = { src.width, src.height,
                    (uchar *)malloc(src.width * src.height * sizeof(uchar)) };
    dilateBinary255(src, tmp, kernel, ksize);
    erodeBinary255(tmp, dst, kernel, ksize);
    delete[] kernel;
    free(tmp.data);
}

void morphClose(s_image src, s_image dst, size_t ksize)
{
    uchar *kernel = getCircleKernel(ksize);
    s_image tmp = { src.width, src.height,
                    (uchar *)malloc(src.width * src.height * sizeof(uchar)) };
    erodeBinary255(src, tmp, kernel, ksize);
    dilateBinary255(tmp, dst, kernel, ksize);
    delete[] kernel;
    free(tmp.data);
}
