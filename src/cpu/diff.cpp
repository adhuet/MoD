#include "mod.hpp"

void diff(s_image src1, s_image src2, s_image dst)
{
    for (int i = 0; i < src1.height * src1.width; i++)
        dst.data[i] = abs(src1.data[i] - src2.data[i]);
}
