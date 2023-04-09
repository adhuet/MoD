#include "utils.hpp"

// FIXME
// Check out Summed Area Tables maybe? Allows to apply filters in constant
// complexity
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

static int squareEuclidDistance(int x1, int y1, int x2, int y2)
{
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
}

uchar *getCircleKernel(size_t diameter)
{
    const int radius = diameter / 2;
    const int square_radius = radius * radius;
    uchar *kernel = new uchar[diameter * diameter];

    for (size_t x = 0; x < diameter; x++)
    {
        for (size_t y = 0; y < diameter; y++)
        {
            int distance = squareEuclidDistance(radius, radius, x, y);
            if (distance >= square_radius)
                kernel[x * diameter + y] = 0;
            else
                kernel[x * diameter + y] = 1;
        }
    }

    return kernel;
}
