#include "utils.hpp"

// FIXME
// Check out Summed Area Tables maybe? Allows to apply filters in constant
// complexity
void filter2D(const SImage &src, SImage &dst, float *kernel, size_t ksize)
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

std::vector<cv::Rect> getBoundingBoxes(int *L, int width, int height)
{
    std::unordered_map<int, struct SBox> boxes;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            // We are on a label
            if (L[y * width + x] != 0)
            {
                int id = L[y * width + x];

                // The box with this label already exists
                if (boxes.count(id) != 0)
                {
                    // Update the SBox attributes
                    boxes[id].max_x = std::max(boxes[id].max_x, x);
                    boxes[id].min_x = std::min(boxes[id].min_x, x);
                    boxes[id].max_y = std::max(boxes[id].max_y, y);
                    boxes[id].min_y = std::min(boxes[id].min_y, y);
                }
                else // Otherwise create it
                    boxes[id] = { x, y, x, y };
            }
        }
    }

    std::vector<cv::Rect> vec;
    for (auto it = boxes.begin(); it != boxes.end(); it++)
        vec.push_back(it->second.toCVRect());
    return vec;
}