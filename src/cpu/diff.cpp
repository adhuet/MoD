#include "mod.hpp"

void diff(const SImage &src1, const SImage &src2, SImage &dst)
{
    for (int i = 0; i < src1.height * src1.width; i++)
        dst.data[i] = abs(src1.data[i] - src2.data[i]);
}
