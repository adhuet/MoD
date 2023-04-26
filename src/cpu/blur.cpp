#include <cmath>
#include <numbers>

#include "mod.hpp"
#include "utils.hpp"

// Try the 2 pass algo (horizontal then vertical)
// Need a convolution
void blur(const SImage &src, SImage &dst, size_t ksize, double sigma)
{
    float *gaussianMatrix = getGaussianMatrix(ksize, sigma);
    filter2D(src, dst, gaussianMatrix, ksize);
    delete[] gaussianMatrix;
}