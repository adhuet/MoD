#include "mod.hpp"

void treshold(s_image src, s_image dst, uchar threshold, uchar maxval)
{
    for (int y = 0; y < src.height; y++)
    {
        for (int x = 0; x < src.width; x++)
        {
            if (src.data[y * src.width + x] < threshold)
                dst.data[y * dst.width + x] = 0;
            else
                dst.data[y * dst.width + x] = maxval;
        }
    }
    return;
}
