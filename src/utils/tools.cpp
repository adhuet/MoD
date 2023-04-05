#include "utils.hpp"

void filter2D(s_image src, s_image dst, float *kernel, size_t ksize)
{
    for (int y = 0; y < src.height; y++)
    {
        for (int x = 0; x < src.width; x++)
        {
            int radius = ksize / 2;
            float sum = 0;
            for (int i = -radius; i <= radius; i++)
            {
                for (int j = -radius; j <= radius; j++)
                {
                    // Skip when out of bounds
                    if (y + i >= 0 && y + i < src.height && x + j >= 0
                        && x + j < src.width)
                        sum += kernel[(i + radius) * ksize + (j + radius)]
                            * src.data[(y + i) * src.width + (x + j)];
                }
            }
            dst.data[y * dst.width + x] = static_cast<uchar>(sum);
        }
    }
}