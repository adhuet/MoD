#include <cmath>
#include <numbers>

#include "mod.hpp"
#include "utils.hpp"

constexpr double pi = 3.14159265358979323846;

static float gaussianFunction(int x, int y, double sigma)
{
    double r = x * x + y * y;
    return exp(-(r / (2 * sigma * sigma))) / (2 * sigma * sigma * pi);
    // return 1.0 / std::sqrt(2 * pi * sigma * sigma)
    //     * std::exp(-(x * x + y * y) / (2 * sigma * sigma));
}

float *getGaussianMatrix(size_t ksize, double sigma)
{
    float *matrix = new float[ksize * ksize];

    int radius = ksize / 2;
    double sum = 0;
    for (int y = -radius; y <= radius; y++)
    {
        for (int x = -radius; x <= radius; x++)
        {
            float kernel_value = gaussianFunction(x, y, sigma);
            matrix[(y + radius) * ksize + (x + radius)] = kernel_value;
            sum += kernel_value;
        }
    }

    for (size_t i = 0; i < ksize; i++)
        for (size_t j = 0; j < ksize; j++)
            matrix[i * ksize + j] /= sum;
    return matrix;
}

// Try the 2 pass algo (horizontal then vertical)
// Need a convolution
void blur(s_image src, s_image dst, size_t ksize, double sigma)
{
    float *gaussianMatrix = getGaussianMatrix(ksize, sigma);
    filter2D(src, dst, gaussianMatrix, ksize);
    delete[] gaussianMatrix;
}