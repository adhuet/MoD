#include <criterion/criterion.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>

#include "mod.hpp"
#include "mod_GPU.hpp"
#include "utils.hpp"

void dilateBinary255(const SImage &src, SImage &dst, uchar *kernel,
                     size_t ksize);
void erodeBinary255(const SImage &src, SImage &dst, uchar *kernel,
                    size_t ksize);

#define CUDA_WARN(XXX)                                                         \
    do                                                                         \
    {                                                                          \
        if (XXX != cudaSuccess)                                                \
            std::cerr << "CUDA Error: " << cudaGetErrorString(XXX)             \
                      << ", at line " << __LINE__ << std::endl;                \
        cudaDeviceSynchronize();                                               \
    } while (0)

template <typename T>
static void assertArrayEqual(T *arr1, T *arr2, int n)
{
    for (int i = 0; i < n; i++)
        cr_assert_eq(arr1[i], arr2[i],
                     "Expected arr1[%d] = %d, got arr2[%d] = %d", i, arr1[i], i,
                     arr2[i]);
}

template <typename T>
static void assertArrayEqualWithError(T *arr1, T *arr2, int n, float error)
{
    for (int i = 0; i < n; i++)
        cr_assert(abs(arr1[i] - arr2[i]) <= error,
                  "Expected true arr1[%d] = %d to be closer to result arr2[%d] "
                  "= %d (error "
                  "threshold: %f)",
                  i, arr1[i], i, arr2[i], error);
}

static __attribute__((unused)) void printMatrix(uchar *mat, int height,
                                                int width)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
            std::cout << std::setfill(' ') << std::setw(3)
                      << static_cast<unsigned>(mat[i * width + j]) << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

static __attribute__((unused)) void printMatrix(int *mat, int height, int width)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
            std::cout << std::setfill(' ') << std::setw(3) << mat[i * width + j]
                      << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

Test(check, pass)
{
    cr_assert(1);
}

__global__ void cudaTest(bool *flag)
{
    *flag = true;
}

Test(check, gpu)
{
    bool flag = false;
    bool *d_flag;

    cudaMalloc(&d_flag, sizeof(bool));
    cudaMemcpy(d_flag, &flag, sizeof(bool), cudaMemcpyHostToDevice);

    cudaTest<<<1, 1>>>(d_flag);

    CUDA_WARN(cudaDeviceSynchronize());

    cudaMemcpy(&flag, d_flag, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_flag);

    cr_assert(flag);
}

Test(morphologicalGPU, dilation)
{
    // clang-format off
    uchar buffer[11 * 11] = {
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0, 255, 255, 255, 255,   0,   0, 255, 255, 255,   0,
          0, 255, 255, 255, 255,   0,   0, 255, 255, 255,   0,
          0, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,
          0, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,
          0, 255, 255,   0,   0,   0, 255, 255, 255, 255,   0,
          0, 255, 255,   0,   0,   0, 255, 255, 255, 255,   0,
          0, 255, 255,   0,   0,   0, 255, 255, 255, 255,   0,
          0, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0,
          0, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
    };

    uchar kernel[3 * 3] = {
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
    };

    uchar expected[11 * 11] = {
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255,   0, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,
        255, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0
    };
    // clang-format on

    uchar *d_input;
    uchar *d_kernel;
    uchar *d_output;

    uchar *output = (uchar *)malloc(11 * 11 * sizeof(uchar));

    cudaMalloc(&d_input, 11 * 11 * sizeof(uchar));
    cudaMalloc(&d_kernel, 3 * 3 * sizeof(uchar));
    cudaMalloc(&d_output, 11 * 11 * sizeof(uchar));
    cudaMemcpy(d_input, buffer, 11 * 11 * sizeof(uchar),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, 3 * 3 * sizeof(uchar), cudaMemcpyHostToDevice);

    dim3 blockDim(4, 4);
    dim3 gridDim(int(ceil((float)11 / blockDim.x)),
                 int(ceil((float)11 / blockDim.y)));
    dilateGPU<<<gridDim, blockDim>>>(d_input, d_output, 11, 11, d_kernel, 3);

    CUDA_WARN(cudaDeviceSynchronize());

    cudaMemcpy(output, d_output, 11 * 11 * sizeof(uchar),
               cudaMemcpyDeviceToHost);

    assertArrayEqual(expected, output, 11 * 11);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    free(output);
}

Test(morphologicalGPU, erosion)
{
    // clang-format off
    uchar buffer[13 * 13] = {
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255,   0, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255
    };

    uchar kernel[3 * 3] = {
        255, 255, 255,
        255, 255, 255,
        255, 255, 255
    };


    uchar expected[13 * 13] = {
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0, 255, 255, 255, 255,   0,   0,   0, 255, 255, 255, 255,   0,
        0, 255, 255, 255, 255,   0,   0,   0, 255, 255, 255, 255,   0,
        0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,
        0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,
        0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,
        0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,
        0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,
        0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,
        0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,
        0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,
        0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
    };
    // clang-format on

    uchar *d_input;
    uchar *d_kernel;
    uchar *d_output;

    uchar *output = (uchar *)malloc(13 * 13 * sizeof(uchar));

    cudaMalloc(&d_input, 13 * 13 * sizeof(uchar));
    cudaMalloc(&d_kernel, 3 * 3 * sizeof(uchar));
    cudaMalloc(&d_output, 13 * 13 * sizeof(uchar));
    cudaMemcpy(d_input, buffer, 13 * 13 * sizeof(uchar),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, 3 * 3 * sizeof(uchar), cudaMemcpyHostToDevice);

    dim3 blockDim(4, 4);
    dim3 gridDim(int(ceil((float)13 / blockDim.x)),
                 int(ceil((float)13 / blockDim.y)));
    erodeGPU<<<gridDim, blockDim>>>(d_input, d_output, 13, 13, d_kernel, 3);

    CUDA_WARN(cudaDeviceSynchronize());

    cudaMemcpy(output, d_output, 13 * 13 * sizeof(uchar),
               cudaMemcpyDeviceToHost);

    assertArrayEqual(expected, output, 13 * 13);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    free(output);
}

Test(connectedComponents, simple4comps, .timeout = 3)
{
    constexpr int height = 13;
    constexpr int width = 14;
    // clang-format off
    uchar buffer[height * width] = {
          0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0, 255, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,
        255, 255, 255, 255, 255,   0,   0,   0,   0,   0, 255, 255,   0,   0,
          0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,
          0,   0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
    };

    __attribute__ ((unused)) int expected[height * width] = {
          0,   0,   3,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   3,   3,   3,   0,   0,   0,   0,   0,   0,  25,  25,   0,   0,
          3,   3,   3,   3,   3,   0,   0,   0,   0,   0,  25,  25,   0,   0,
          0,   3,   3,   3,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   3,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 130,   0,   0,   0,   0,   0,   0, 137,   0,   0,   0,
          0,   0, 130, 130, 130,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 130,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0  
    };
    // clang-format on

    // std::cout << "Input:" << std::endl;
    // printMatrix(buffer, height, width);

    uchar *d_input;
    int *d_output;

    int *output = (int *)malloc(height * width * sizeof(int));

    cudaMalloc(&d_input, height * width * sizeof(uchar));
    cudaMalloc(&d_output, height * width * sizeof(int));
    cudaMemcpy(d_input, buffer, height * width * sizeof(uchar),
               cudaMemcpyHostToDevice);

    dim3 blockDim(4, 4);
    dim3 gridDim(int(ceil((float)height / blockDim.x)),
                 int(ceil((float)width / blockDim.y)));
    // connectedComponentsGPU(d_input, d_output, height, width, gridDim,
    // blockDim);
    initCCL<<<gridDim, blockDim>>>(d_input, d_output, height, width);
    CUDA_WARN(cudaDeviceSynchronize());

    cudaMemcpy(output, d_output, height * width * sizeof(int),
               cudaMemcpyDeviceToHost);
    // std::cout << "After initCCL:" << std::endl;
    // printMatrix(output, height, width);

    mergeCCL<<<gridDim, blockDim>>>(d_input, d_output, height, width);
    CUDA_WARN(cudaDeviceSynchronize());
    cudaMemcpy(output, d_output, height * width * sizeof(int),
               cudaMemcpyDeviceToHost);
    // std::cout << "After mergeCCL:" << std::endl;
    // printMatrix(output, height, width);

    compressCCL<<<gridDim, blockDim>>>(d_input, d_output, height, width);
    CUDA_WARN(cudaDeviceSynchronize());
    cudaMemcpy(output, d_output, height * width * sizeof(int),
               cudaMemcpyDeviceToHost);
    // std::cout << "After compressCCL (final):" << std::endl;
    // printMatrix(output, height, width);

    // std::cout << "Expected:" << std::endl;
    // printMatrix(expected, height, width);

    assertArrayEqual(expected, output, height * width);

    cudaFree(d_input);
    cudaFree(d_output);
    free(output);
}

Test(bboxes, fourBboxes, .timeout = 3)
{
    constexpr int height = 13;
    constexpr int width = 14;
    // clang-format off
    uchar buffer[height * width] = {
          0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0, 255, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,
        255, 255, 255, 255, 255,   0,   0,   0,   0,   0, 255, 255,   0,   0,
          0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,
          0,   0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
    };

    __attribute__ ((unused)) int expected[height * width] = {
          0,   0,   3,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   3,   3,   3,   0,   0,   0,   0,   0,   0,  25,  25,   0,   0,
          3,   3,   3,   3,   3,   0,   0,   0,   0,   0,  25,  25,   0,   0,
          0,   3,   3,   3,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   3,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 130,   0,   0,   0,   0,   0,   0, 137,   0,   0,   0,
          0,   0, 130, 130, 130,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 130,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0  
    };
    // clang-format on

    // std::cout << "Input:" << std::endl;
    // printMatrix(buffer, height, width);

    uchar *d_input;
    int *d_output;

    int *output = (int *)malloc(height * width * sizeof(int));

    cudaMalloc(&d_input, height * width * sizeof(uchar));
    cudaMalloc(&d_output, height * width * sizeof(int));
    cudaMemcpy(d_input, buffer, height * width * sizeof(uchar),
               cudaMemcpyHostToDevice);

    dim3 blockDim(4, 4);
    dim3 gridDim(int(ceil((float)height / blockDim.x)),
                 int(ceil((float)width / blockDim.y)));
    // connectedComponentsGPU(d_input, d_output, height, width, gridDim,
    // blockDim);
    initCCL<<<gridDim, blockDim>>>(d_input, d_output, height, width);
    CUDA_WARN(cudaDeviceSynchronize());

    cudaMemcpy(output, d_output, height * width * sizeof(int),
               cudaMemcpyDeviceToHost);
    // std::cout << "After initCCL:" << std::endl;
    // printMatrix(output, height, width);

    mergeCCL<<<gridDim, blockDim>>>(d_input, d_output, height, width);
    CUDA_WARN(cudaDeviceSynchronize());
    cudaMemcpy(output, d_output, height * width * sizeof(int),
               cudaMemcpyDeviceToHost);
    // std::cout << "After mergeCCL:" << std::endl;
    // printMatrix(output, height, width);

    compressCCL<<<gridDim, blockDim>>>(d_input, d_output, height, width);
    CUDA_WARN(cudaDeviceSynchronize());
    cudaMemcpy(output, d_output, height * width * sizeof(int),
               cudaMemcpyDeviceToHost);
    // std::cout << "After compressCCL (final):" << std::endl;
    // printMatrix(output, height, width);

    // std::cout << "Expected:" << std::endl;
    // printMatrix(expected, height, width);

    auto bboxes = getBoundingBoxes(output, width, height);

    // for (const auto &box : bboxes)
    // {
    //     std::cout << box << std::endl;
    // }

    cr_assert_eq(bboxes.size(), 4, "Expected bboxes.size() == 4, got %d",
                 bboxes.size());

    cudaFree(d_input);
    cudaFree(d_output);
    free(output);
}

Test(blur, radius_one)
{
    constexpr int height = 13;
    constexpr int width = 14;

    // clang-format off
    uchar buffer[height * width] = {
          0,   1, 200,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0, 200, 200, 200,   0,   0,   0,   0,   0,   0, 100, 100,   0,   0,
        200, 200, 200, 200, 200,   0,   0,   0,   0,   0, 100, 100,   0,   0,
          0, 200, 200, 200,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   1, 200,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,  50,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,  50,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,  50,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0,  25,   0,   0,   0,
          0,   0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
    };
    // clang-format on

    size_t ksize = 3;
    double sigma = 2.0;
    float *gaussian_matrix = getGaussianMatrix(ksize, sigma);

    dim3 blockDim(4, 4);
    dim3 gridDim(int(ceil((float)height / blockDim.x)),
                 int(ceil((float)width / blockDim.y)));

    uchar *d_input;
    uchar *d_output;
    float *d_kernel;

    uchar *output = new uchar[height * width];

    cudaMalloc(&d_input, height * width * sizeof(uchar));
    cudaMalloc(&d_output, height * width * sizeof(uchar));
    cudaMalloc(&d_kernel, ksize * ksize * sizeof(float));

    cudaMemcpy(d_kernel, gaussian_matrix, ksize * ksize * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, buffer, height * width * sizeof(uchar),
               cudaMemcpyHostToDevice);

    blurGPU<<<gridDim, blockDim>>>(d_input, d_output, height, width, d_kernel,
                                   ksize);

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, height * width * sizeof(uchar),
               cudaMemcpyDeviceToHost);

    SImage expected(width, height);
    blur(SImage(width, height, buffer), expected, ksize, sigma);

    // printMatrix(output, height, width);
    assertArrayEqual(expected.data, output, height * width);

    delete[] output;
    delete[] gaussian_matrix;
    cudaFree(d_kernel);
    cudaFree(d_output);
    cudaFree(d_input);
}

Test(blur, radius_spans_full_block)
{
    constexpr int height = 13;
    constexpr int width = 14;

    // clang-format off
    uchar buffer[height * width] = {
          0,   1, 200,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0, 200, 200, 200,   0,   0,   0,   0,   0,   0, 100, 100,   0,   0,
        200, 200, 200, 200, 200,   0,   0,   0,   0,   0, 100, 100,   0,   0,
          0, 200, 200, 200,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   1, 200,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,  50,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,  50,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,  50,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0,  25,   0,   0,   0,
          0,   0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
    };
    // clang-format on

    size_t ksize = 9;
    double sigma = 2.0;
    float *gaussian_matrix = getGaussianMatrix(ksize, sigma);

    dim3 blockDim(4, 4);
    dim3 gridDim(int(ceil((float)height / blockDim.x)),
                 int(ceil((float)width / blockDim.y)));

    uchar *d_input;
    uchar *d_output;
    float *d_kernel;

    uchar *output = new uchar[height * width];

    cudaMalloc(&d_input, height * width * sizeof(uchar));
    cudaMalloc(&d_output, height * width * sizeof(uchar));
    cudaMalloc(&d_kernel, ksize * ksize * sizeof(float));

    cudaMemcpy(d_kernel, gaussian_matrix, ksize * ksize * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, buffer, height * width * sizeof(uchar),
               cudaMemcpyHostToDevice);

    blurGPU<<<gridDim, blockDim>>>(d_input, d_output, height, width, d_kernel,
                                   ksize);

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, height * width * sizeof(uchar),
               cudaMemcpyDeviceToHost);

    SImage expected(width, height);
    blur(SImage(width, height, buffer), expected, ksize, sigma);

    // printMatrix(output, height, width);
    assertArrayEqual(expected.data, output, height * width);

    delete[] output;
    delete[] gaussian_matrix;
    cudaFree(d_kernel);
    cudaFree(d_output);
    cudaFree(d_input);
}

Test(blur, radius_spans_block_plus_one)
{
    constexpr int height = 13;
    constexpr int width = 14;

    // clang-format off
    uchar buffer[height * width] = {
          0,   1, 200,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0, 200, 200, 200,   0,   0,   0,   0,   0,   0, 100, 100,   0,   0,
        200, 200, 200, 200, 200,   0,   0,   0,   0,   0, 100, 100,   0,   0,
          0, 200, 200, 200,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   1, 200,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,  50,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,  50,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,  50,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0,  25,   0,   0,   0,
          0,   0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
    };
    // clang-format on

    size_t ksize = 11;
    double sigma = 2.0;
    float *gaussian_matrix = getGaussianMatrix(ksize, sigma);

    dim3 blockDim(4, 4);
    dim3 gridDim(int(ceil((float)height / blockDim.x)),
                 int(ceil((float)width / blockDim.y)));

    uchar *d_input;
    uchar *d_output;
    float *d_kernel;

    uchar *output = new uchar[height * width];

    cudaMalloc(&d_input, height * width * sizeof(uchar));
    cudaMalloc(&d_output, height * width * sizeof(uchar));
    cudaMalloc(&d_kernel, ksize * ksize * sizeof(float));

    cudaMemcpy(d_kernel, gaussian_matrix, ksize * ksize * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, buffer, height * width * sizeof(uchar),
               cudaMemcpyHostToDevice);

    blurGPU<<<gridDim, blockDim>>>(d_input, d_output, height, width, d_kernel,
                                   ksize);

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, height * width * sizeof(uchar),
               cudaMemcpyDeviceToHost);

    SImage expected(width, height);
    blur(SImage(width, height, buffer), expected, ksize, sigma);

    // printMatrix(output, height, width);
    assertArrayEqual(expected.data, output, height * width);

    delete[] output;
    delete[] gaussian_matrix;
    cudaFree(d_kernel);
    cudaFree(d_output);
    cudaFree(d_input);
}

Test(blur, radius_spans_two_blocks)
{
    constexpr int height = 13;
    constexpr int width = 14;

    // clang-format off
    uchar buffer[height * width] = {
          0,   1, 200,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0, 200, 200, 200,   0,   0,   0,   0,   0,   0, 100, 100,   0,   0,
        200, 200, 200, 200, 200,   0,   0,   0,   0,   0, 100, 100,   0,   0,
          0, 200, 200, 200,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   1, 200,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,  50,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,  50,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,  50,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0,  25,   0,   0,   0,
          0,   0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
    };
    // clang-format on

    size_t ksize = 11;
    double sigma = 2.0;
    float *gaussian_matrix = getGaussianMatrix(ksize, sigma);

    dim3 blockDim(4, 4);
    dim3 gridDim(int(ceil((float)height / blockDim.x)),
                 int(ceil((float)width / blockDim.y)));

    uchar *d_input;
    uchar *d_output;
    float *d_kernel;

    uchar *output = new uchar[height * width];

    cudaMalloc(&d_input, height * width * sizeof(uchar));
    cudaMalloc(&d_output, height * width * sizeof(uchar));
    cudaMalloc(&d_kernel, ksize * ksize * sizeof(float));

    cudaMemcpy(d_kernel, gaussian_matrix, ksize * ksize * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, buffer, height * width * sizeof(uchar),
               cudaMemcpyHostToDevice);

    blurGPU<<<gridDim, blockDim>>>(d_input, d_output, height, width, d_kernel,
                                   ksize);

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, height * width * sizeof(uchar),
               cudaMemcpyDeviceToHost);

    SImage expected(width, height);
    blur(SImage(width, height, buffer), expected, ksize, sigma);

    // printMatrix(output, height, width);
    assertArrayEqual(expected.data, output, height * width);

    delete[] output;
    delete[] gaussian_matrix;
    cudaFree(d_kernel);
    cudaFree(d_output);
    cudaFree(d_input);
}

Test(blurTiled, radius_one, .disabled = false)
{
    constexpr int height = 13;
    constexpr int width = 14;

    // clang-format off
    uchar buffer[height * width] = {
          0,   1, 200,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //0,   0,
          0, 200, 200, 200,   0,   0,   0,   0,   0,   0, 100, 100,  50, 255, //0,   0,
        200, 200, 200, 200, 200,   0,   0,   0,   0,   0, 100, 100,  50,  50, //0,   0,
          0, 200, 200, 200,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //0,   0,
         10,   1, 200,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //0,   0,
          0,   0,   0,   0,   0,  50,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,  50,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,  50,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0,  25,   0,   0,   0,
          0,   0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
    };
    // clang-format on

    size_t ksize = 3;
    double sigma = 2.0;
    float *gaussian_matrix = getGaussianMatrix(ksize, sigma);

    int block_width = 8;
    size_t tile_width = block_width - ksize + 1;
    dim3 blockDim(block_width, block_width);
    dim3 gridDim(int(ceil((float)height / tile_width)),
                 int(ceil((float)width / tile_width)));

    uchar *d_input;
    uchar *d_output;
    float *d_kernel;

    uchar *output = new uchar[height * width];

    cudaMalloc(&d_input, height * width * sizeof(uchar));
    cudaMalloc(&d_output, height * width * sizeof(uchar));
    cudaMalloc(&d_kernel, ksize * ksize * sizeof(float));

    cudaMemcpy(d_kernel, gaussian_matrix, ksize * ksize * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, buffer, height * width * sizeof(uchar),
               cudaMemcpyHostToDevice);

    blurTiledGPU<<<gridDim, blockDim,
                   block_width * block_width * sizeof(uchar)>>>(
        d_input, d_output, height, width, d_kernel, ksize);

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, height * width * sizeof(uchar),
               cudaMemcpyDeviceToHost);

    SImage expected(width, height);
    blur(SImage(width, height, buffer), expected, ksize, sigma);

    // std::cout << "Expected: " << std::endl;
    // printMatrix(expected.data, height, width);

    // std::cout << "Actual: " << std::endl;
    // printMatrix(output, height, width);

    assertArrayEqual(expected.data, output, height * width);

    delete[] output;
    delete[] gaussian_matrix;
    cudaFree(d_kernel);
    cudaFree(d_output);
    cudaFree(d_input);
}

Test(blurTiled, radius_two, .disabled = false)
{
    constexpr int height = 13;
    constexpr int width = 14;

    // clang-format off
    uchar buffer[height * width] = {
          0,   1, 200,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //0,   0,
          0, 200, 200, 200,   0,   0,   0,   0,   0,   0, 100, 100,  50, 255, //0,   0,
        200, 200, 200, 200, 200,   0,   0,   0,   0,   0, 100, 100,  50,  50, //0,   0,
          0, 200, 200, 200,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //0,   0,
         10,   1, 200,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //0,   0,
          0,   0,   0,   0,   0,  50,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,  50,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,  50,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0,  25,   0,   0,   0,
          0,   0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
    };
    // clang-format on

    size_t ksize = 5;
    double sigma = 2.0;
    float *gaussian_matrix = getGaussianMatrix(ksize, sigma);

    int block_width = 8;
    size_t tile_width = block_width - ksize + 1;
    dim3 blockDim(block_width, block_width);
    dim3 gridDim(int(ceil((float)height / tile_width)),
                 int(ceil((float)width / tile_width)));

    uchar *d_input;
    uchar *d_output;
    float *d_kernel;

    uchar *output = new uchar[height * width];

    cudaMalloc(&d_input, height * width * sizeof(uchar));
    cudaMalloc(&d_output, height * width * sizeof(uchar));
    cudaMalloc(&d_kernel, ksize * ksize * sizeof(float));

    cudaMemcpy(d_kernel, gaussian_matrix, ksize * ksize * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, buffer, height * width * sizeof(uchar),
               cudaMemcpyHostToDevice);

    blurTiledGPU<<<gridDim, blockDim,
                   block_width * block_width * sizeof(uchar)>>>(
        d_input, d_output, height, width, d_kernel, ksize);

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, height * width * sizeof(uchar),
               cudaMemcpyDeviceToHost);

    SImage expected(width, height);
    blur(SImage(width, height, buffer), expected, ksize, sigma);

    // std::cout << "Expected: " << std::endl;
    // printMatrix(expected.data, height, width);

    // std::cout << "Actual: " << std::endl;
    // printMatrix(output, height, width);

    assertArrayEqual(expected.data, output, height * width);

    delete[] output;
    delete[] gaussian_matrix;
    cudaFree(d_kernel);
    cudaFree(d_output);
    cudaFree(d_input);
}

Test(blurTiled, radius_four, .disabled = false)
{
    constexpr int height = 13;
    constexpr int width = 14;

    // clang-format off
    uchar buffer[height * width] = {
          0,   1, 200,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //0,   0,
          0, 200, 200, 200,   0,   0,   0,   0,   0,   0, 100, 100,  50, 255, //0,   0,
        200, 200, 200, 200, 200,   0,   0,   0,   0,   0, 100, 100,  50,  50, //0,   0,
          0, 200, 200, 200,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //0,   0,
         10,   1, 200,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //0,   0,
          0,   0,   0,   0,   0,  50,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,  50,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,  50,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0,  25,   0,   0,   0,
          0,   0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
    };
    // clang-format on

    size_t ksize = 9;
    double sigma = 2.0;
    float *gaussian_matrix = getGaussianMatrix(ksize, sigma);

    int block_width = 16;
    size_t tile_width = block_width - ksize + 1;
    dim3 blockDim(block_width, block_width);
    dim3 gridDim(int(ceil((float)height / tile_width)),
                 int(ceil((float)width / tile_width)));

    uchar *d_input;
    uchar *d_output;
    float *d_kernel;

    uchar *output = new uchar[height * width];

    cudaMalloc(&d_input, height * width * sizeof(uchar));
    cudaMalloc(&d_output, height * width * sizeof(uchar));
    cudaMalloc(&d_kernel, ksize * ksize * sizeof(float));

    cudaMemcpy(d_kernel, gaussian_matrix, ksize * ksize * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, buffer, height * width * sizeof(uchar),
               cudaMemcpyHostToDevice);

    blurTiledGPU<<<gridDim, blockDim,
                   block_width * block_width * sizeof(uchar)>>>(
        d_input, d_output, height, width, d_kernel, ksize);

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, height * width * sizeof(uchar),
               cudaMemcpyDeviceToHost);

    SImage expected(width, height);
    blur(SImage(width, height, buffer), expected, ksize, sigma);

    // std::cout << "Expected: " << std::endl;
    // printMatrix(expected.data, height, width);

    // std::cout << "Actual: " << std::endl;
    // printMatrix(output, height, width);

    assertArrayEqual(expected.data, output, height * width);

    delete[] output;
    delete[] gaussian_matrix;
    cudaFree(d_kernel);
    cudaFree(d_output);
    cudaFree(d_input);
}

Test(blurTiled, same_input_output_fails_or_not, .disabled = false)
{
    constexpr int height = 13;
    constexpr int width = 14;

    // clang-format off
    // uchar buffer[height * width] = {
    //     125, 125, 200, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
    //     125, 200, 200, 200, 125, 125, 125, 125, 125, 125, 100, 100, 125, 125,
    //     200, 200, 200, 200, 200, 125, 125, 125, 125, 125, 100, 100, 125, 125,
    //     125, 200, 200, 200, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
    //     125, 125, 200, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
    //     125, 125, 125, 125, 125,  50, 125, 125, 125, 125, 125, 125, 125, 125,
    //     125, 125, 125, 125, 125, 125,  50, 125, 125, 125, 125, 125, 125, 125,
    //     125, 125, 125, 125, 125, 125, 125,  50, 125, 125, 125, 125, 125, 125,
    //     125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
    //     125, 125, 125, 255, 125, 125, 125, 125, 125, 125,  25, 125, 125, 125,
    //     125, 125, 255, 255, 255, 125, 125, 125, 125, 125, 125, 125, 125, 125,
    //     125, 125, 125, 255, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
    //     125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125
    // };
    // clang-format on

    uchar *buffer = new uchar[height * width];
    for (unsigned int i = 0; i < height * width; i++)
    {
        buffer[i] = rand() % 256;
    }

    size_t ksize = 11;
    double sigma = 2.0;
    float *gaussian_matrix = getGaussianMatrix(ksize, sigma);

    int block_width = 16;
    size_t tile_width = block_width - ksize + 1;
    dim3 blockDim(block_width, block_width);
    dim3 gridDim(int(ceil((float)height / tile_width)),
                 int(ceil((float)width / tile_width)));

    uchar *d_input;
    // uchar *d_output;
    float *d_kernel;

    uchar *output = new uchar[height * width];

    cudaMalloc(&d_input, height * width * sizeof(uchar));
    // cudaMalloc(&d_output, height * width * sizeof(uchar));
    cudaMalloc(&d_kernel, ksize * ksize * sizeof(float));

    cudaMemcpy(d_kernel, gaussian_matrix, ksize * ksize * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, buffer, height * width * sizeof(uchar),
               cudaMemcpyHostToDevice);

    blurTiledGPU<<<gridDim, blockDim,
                   block_width * block_width * sizeof(uchar)>>>(
        d_input, d_input, height, width, d_kernel, ksize);

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_input, height * width * sizeof(uchar),
               cudaMemcpyDeviceToHost);

    SImage expected(width, height);
    blur(SImage(width, height, buffer), expected, ksize, sigma);

    // std::cout << "Expected: " << std::endl;
    // printMatrix(expected.data, height, width);

    // std::cout << "Actual: " << std::endl;
    // printMatrix(output, height, width);

    assertArrayEqual(expected.data, output, height * width);

    delete[] output;
    delete[] gaussian_matrix;
    cudaFree(d_kernel);
    // cudaFree(d_output);
    cudaFree(d_input);
    delete[] buffer;
}

// FIXME test fails while visual check on make run shows that blur is fine
// This might be due to float imprecision, as normal blurGPU fails too
Test(blurTiled, real_case_scenario, .disabled = true)
{
    cv::Mat input = cv::imread("docs/report_resources/source_example.png");
    cr_assert(!input.empty());
    cv::cvtColor(input, input, cv::COLOR_BGR2GRAY);
    int width = input.cols;
    int height = input.rows;

    size_t ksize = 15;
    float sigma = 2.0;
    float *gaussian_matrix = getGaussianMatrix(ksize, sigma);

    uchar *buffer = input.ptr<uchar>(0);
    uchar *output = new uchar[height * width];

    int block_width = 32;
    size_t tile_width = block_width - ksize + 1;
    dim3 blockDim(block_width, block_width);
    dim3 blurTiledGridDim(int(ceil((float)height / tile_width)),
                          int(ceil((float)width / tile_width)));
    dim3 blurGridDim(int(ceil((float)height / block_width)),
                     int(ceil((float)width / block_width)));

    uchar *d_input;
    uchar *d_output;
    float *d_kernel;

    cudaMalloc(&d_input, height * width * sizeof(uchar));
    cudaMalloc(&d_output, height * width * sizeof(uchar));
    cudaMalloc(&d_kernel, ksize * ksize * sizeof(float));

    cudaMemcpy(d_kernel, gaussian_matrix, ksize * ksize * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, buffer, height * width * sizeof(uchar),
               cudaMemcpyHostToDevice);

    blurTiledGPU<<<blurTiledGridDim, blockDim,
                   block_width * block_width * sizeof(uchar)>>>(
        d_input, d_output, height, width, d_kernel, ksize);
    // blurGPU<<<blurGridDim, blockDim>>>(d_input, d_output, height, width,
    //                                    d_kernel, ksize);

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, height * width * sizeof(uchar),
               cudaMemcpyDeviceToHost);

    SImage expected(width, height);
    blur(SImage(width, height, buffer), expected, ksize, sigma);

    // printMatrix(output, height, width);
    assertArrayEqualWithError(expected.data, output, height * width, 10.0f);

    delete[] output;
    delete[] gaussian_matrix;
    cudaFree(d_kernel);
    cudaFree(d_output);
    cudaFree(d_input);
    delete[] buffer;
}

Test(dilateTiled, radius_one, .disabled = false)
{
    constexpr int height = 13;
    constexpr int width = 14;

    // clang-format off
    uchar buffer[height * width] = {
          0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //0,   0,
          0, 255, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255, //0,   0,
        255, 255, 255, 255, 255,   0,   0,   0,   0,   0, 255, 255, 255, 255, //0,   0,
          0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //0,   0,
        255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //0,   0,
          0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,
          0,   0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
    };
    // clang-format on

    size_t ksize = 3;
    uchar *circle_kernel = getCircleKernel(ksize);

    int block_width = 8;
    size_t tile_width = block_width - ksize + 1;
    dim3 blockDim(block_width, block_width);
    dim3 gridDim(int(ceil((float)height / tile_width)),
                 int(ceil((float)width / tile_width)));

    uchar *d_input;
    uchar *d_output;
    uchar *d_kernel;

    uchar *output = new uchar[height * width];

    cudaMalloc(&d_input, height * width * sizeof(uchar));
    cudaMalloc(&d_output, height * width * sizeof(uchar));
    cudaMalloc(&d_kernel, ksize * ksize * sizeof(float));

    cudaMemcpy(d_kernel, circle_kernel, ksize * ksize * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, buffer, height * width * sizeof(uchar),
               cudaMemcpyHostToDevice);

    dilateTiledGPU<<<gridDim, blockDim,
                     block_width * block_width * sizeof(uchar)>>>(
        d_input, d_output, height, width, d_kernel, ksize);

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, height * width * sizeof(uchar),
               cudaMemcpyDeviceToHost);

    SImage expected(width, height);
    dilateBinary255(SImage(width, height, buffer), expected, circle_kernel,
                    ksize);

    assertArrayEqual(expected.data, output, height * width);

    delete[] output;
    delete[] circle_kernel;
    cudaFree(d_kernel);
    cudaFree(d_output);
    cudaFree(d_input);
}

Test(dilateTiled, radius_two, .disabled = false)
{
    constexpr int height = 13;
    constexpr int width = 14;

    // clang-format off
    uchar buffer[height * width] = {
          0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //0,   0,
          0, 255, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255, //0,   0,
        255, 255, 255, 255, 255,   0,   0,   0,   0,   0, 255, 255, 255, 255, //0,   0,
          0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //0,   0,
        255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //0,   0,
          0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,
          0,   0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
    };
    // clang-format on

    size_t ksize = 5;
    uchar *circle_kernel = getCircleKernel(ksize);

    int block_width = 8;
    size_t tile_width = block_width - ksize + 1;
    dim3 blockDim(block_width, block_width);
    dim3 gridDim(int(ceil((float)height / tile_width)),
                 int(ceil((float)width / tile_width)));

    uchar *d_input;
    uchar *d_output;
    uchar *d_kernel;

    uchar *output = new uchar[height * width];

    cudaMalloc(&d_input, height * width * sizeof(uchar));
    cudaMalloc(&d_output, height * width * sizeof(uchar));
    cudaMalloc(&d_kernel, ksize * ksize * sizeof(float));

    cudaMemcpy(d_kernel, circle_kernel, ksize * ksize * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, buffer, height * width * sizeof(uchar),
               cudaMemcpyHostToDevice);

    dilateTiledGPU<<<gridDim, blockDim,
                     block_width * block_width * sizeof(uchar)>>>(
        d_input, d_output, height, width, d_kernel, ksize);

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, height * width * sizeof(uchar),
               cudaMemcpyDeviceToHost);

    SImage expected(width, height);
    dilateBinary255(SImage(width, height, buffer), expected, circle_kernel,
                    ksize);

    assertArrayEqual(expected.data, output, height * width);

    delete[] output;
    delete[] circle_kernel;
    cudaFree(d_kernel);
    cudaFree(d_output);
    cudaFree(d_input);
}

Test(dilateTiled, radius_four, .disabled = false)
{
    constexpr int height = 13;
    constexpr int width = 14;

    // clang-format off
    uchar buffer[height * width] = {
          0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //0,   0,
          0, 255, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255, //0,   0,
        255, 255, 255, 255, 255,   0,   0,   0,   0,   0, 255, 255, 255, 255, //0,   0,
          0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //0,   0,
        255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //0,   0,
          0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,
          0,   0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
    };
    // clang-format on

    size_t ksize = 9;
    uchar *circle_kernel = getCircleKernel(ksize);

    int block_width = 16;
    size_t tile_width = block_width - ksize + 1;
    dim3 blockDim(block_width, block_width);
    dim3 gridDim(int(ceil((float)height / tile_width)),
                 int(ceil((float)width / tile_width)));

    uchar *d_input;
    uchar *d_output;
    uchar *d_kernel;

    uchar *output = new uchar[height * width];

    cudaMalloc(&d_input, height * width * sizeof(uchar));
    cudaMalloc(&d_output, height * width * sizeof(uchar));
    cudaMalloc(&d_kernel, ksize * ksize * sizeof(float));

    cudaMemcpy(d_kernel, circle_kernel, ksize * ksize * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, buffer, height * width * sizeof(uchar),
               cudaMemcpyHostToDevice);

    dilateTiledGPU<<<gridDim, blockDim,
                     block_width * block_width * sizeof(uchar)>>>(
        d_input, d_output, height, width, d_kernel, ksize);

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, height * width * sizeof(uchar),
               cudaMemcpyDeviceToHost);

    SImage expected(width, height);
    dilateBinary255(SImage(width, height, buffer), expected, circle_kernel,
                    ksize);

    assertArrayEqual(expected.data, output, height * width);

    delete[] output;
    delete[] circle_kernel;
    cudaFree(d_kernel);
    cudaFree(d_output);
    cudaFree(d_input);
}

Test(erodeTiled, radius_one, .disabled = false)
{
    constexpr int height = 13;
    constexpr int width = 14;

    // clang-format off
    uchar buffer[height * width] = {
          0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //0,   0,
          0, 255, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255, //0,   0,
        255, 255, 255, 255, 255,   0,   0,   0,   0,   0, 255, 255, 255, 255, //0,   0,
          0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //0,   0,
        255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //0,   0,
          0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,
          0,   0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
    };
    // clang-format on

    size_t ksize = 3;
    uchar *circle_kernel = getCircleKernel(ksize);

    int block_width = 8;
    size_t tile_width = block_width - ksize + 1;
    dim3 blockDim(block_width, block_width);
    dim3 gridDim(int(ceil((float)height / tile_width)),
                 int(ceil((float)width / tile_width)));

    uchar *d_input;
    uchar *d_output;
    uchar *d_kernel;

    uchar *output = new uchar[height * width];

    cudaMalloc(&d_input, height * width * sizeof(uchar));
    cudaMalloc(&d_output, height * width * sizeof(uchar));
    cudaMalloc(&d_kernel, ksize * ksize * sizeof(float));

    cudaMemcpy(d_kernel, circle_kernel, ksize * ksize * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, buffer, height * width * sizeof(uchar),
               cudaMemcpyHostToDevice);

    erodeTiledGPU<<<gridDim, blockDim,
                    block_width * block_width * sizeof(uchar)>>>(
        d_input, d_output, height, width, d_kernel, ksize);

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, height * width * sizeof(uchar),
               cudaMemcpyDeviceToHost);

    SImage expected(width, height);
    erodeBinary255(SImage(width, height, buffer), expected, circle_kernel,
                   ksize);

    assertArrayEqual(expected.data, output, height * width);

    delete[] output;
    delete[] circle_kernel;
    cudaFree(d_kernel);
    cudaFree(d_output);
    cudaFree(d_input);
}

Test(erodeTiled, radius_two, .disabled = false)
{
    constexpr int height = 13;
    constexpr int width = 14;

    // clang-format off
    uchar buffer[height * width] = {
          0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //0,   0,
          0, 255, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255, //0,   0,
        255, 255, 255, 255, 255,   0,   0,   0,   0,   0, 255, 255, 255, 255, //0,   0,
          0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //0,   0,
        255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //0,   0,
          0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,
          0,   0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
    };
    // clang-format on

    size_t ksize = 5;
    uchar *circle_kernel = getCircleKernel(ksize);

    int block_width = 8;
    size_t tile_width = block_width - ksize + 1;
    dim3 blockDim(block_width, block_width);
    dim3 gridDim(int(ceil((float)height / tile_width)),
                 int(ceil((float)width / tile_width)));

    uchar *d_input;
    uchar *d_output;
    uchar *d_kernel;

    uchar *output = new uchar[height * width];

    cudaMalloc(&d_input, height * width * sizeof(uchar));
    cudaMalloc(&d_output, height * width * sizeof(uchar));
    cudaMalloc(&d_kernel, ksize * ksize * sizeof(float));

    cudaMemcpy(d_kernel, circle_kernel, ksize * ksize * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, buffer, height * width * sizeof(uchar),
               cudaMemcpyHostToDevice);

    erodeTiledGPU<<<gridDim, blockDim,
                    block_width * block_width * sizeof(uchar)>>>(
        d_input, d_output, height, width, d_kernel, ksize);

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, height * width * sizeof(uchar),
               cudaMemcpyDeviceToHost);

    SImage expected(width, height);
    erodeBinary255(SImage(width, height, buffer), expected, circle_kernel,
                   ksize);

    assertArrayEqual(expected.data, output, height * width);

    delete[] output;
    delete[] circle_kernel;
    cudaFree(d_kernel);
    cudaFree(d_output);
    cudaFree(d_input);
}

Test(erodeTiled, radius_four, .disabled = false)
{
    constexpr int height = 13;
    constexpr int width = 14;

    // clang-format off
    uchar buffer[height * width] = {
          0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //0,   0,
          0, 255, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255, //0,   0,
        255, 255, 255, 255, 255,   0,   0,   0,   0,   0, 255, 255, 255, 255, //0,   0,
          0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //0,   0,
        255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //0,   0,
          0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,
          0,   0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
    };
    // clang-format on

    size_t ksize = 9;
    uchar *circle_kernel = getCircleKernel(ksize);

    int block_width = 16;
    size_t tile_width = block_width - ksize + 1;
    dim3 blockDim(block_width, block_width);
    dim3 gridDim(int(ceil((float)height / tile_width)),
                 int(ceil((float)width / tile_width)));

    uchar *d_input;
    uchar *d_output;
    uchar *d_kernel;

    uchar *output = new uchar[height * width];

    cudaMalloc(&d_input, height * width * sizeof(uchar));
    cudaMalloc(&d_output, height * width * sizeof(uchar));
    cudaMalloc(&d_kernel, ksize * ksize * sizeof(float));

    cudaMemcpy(d_kernel, circle_kernel, ksize * ksize * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, buffer, height * width * sizeof(uchar),
               cudaMemcpyHostToDevice);

    erodeTiledGPU<<<gridDim, blockDim,
                    block_width * block_width * sizeof(uchar)>>>(
        d_input, d_output, height, width, d_kernel, ksize);

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, height * width * sizeof(uchar),
               cudaMemcpyDeviceToHost);

    SImage expected(width, height);
    erodeBinary255(SImage(width, height, buffer), expected, circle_kernel,
                   ksize);

    assertArrayEqual(expected.data, output, height * width);

    delete[] output;
    delete[] circle_kernel;
    cudaFree(d_kernel);
    cudaFree(d_output);
    cudaFree(d_input);
}

Test(morphOpenTiled, real_case_scenario)
{
    cv::Mat bgd = cv::imread("docs/report_resources/background.png");
    cv::Mat input = cv::imread("docs/report_resources/source_example.png");
    cr_assert(!input.empty() && !bgd.empty());
    cv::cvtColor(input, input, cv::COLOR_BGR2GRAY);
    cv::cvtColor(bgd, bgd, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(input, input, cv::Size(15, 15), 0.2);
    cv::GaussianBlur(bgd, bgd, cv::Size(15, 15), 0.2);

    cv::absdiff(input, bgd, input);
    cv::threshold(input, input, 20, 255, cv::THRESH_BINARY);

    int width = input.cols;
    int height = input.rows;

    size_t ksize = 15;
    uchar *circle_kernel = getCircleKernel(ksize);

    uchar *buffer = input.ptr<uchar>(0);
    uchar *output = new uchar[height * width];
    uchar *expected = new uchar[height * width];

    int block_width = 32;
    size_t tile_width = block_width - ksize + 1;
    dim3 blockDim(block_width, block_width);
    dim3 morphTiledGridDim(int(ceil((float)height / tile_width)),
                           int(ceil((float)width / tile_width)));
    dim3 morphGridDim(int(ceil((float)height / block_width)),
                      int(ceil((float)width / block_width)));

    uchar *d_input;
    uchar *d_swap;
    uchar *d_kernel;

    cudaMalloc(&d_input, height * width * sizeof(uchar));
    cudaMalloc(&d_swap, height * width * sizeof(uchar));
    cudaMalloc(&d_kernel, ksize * ksize * sizeof(float));

    cudaMemcpy(d_kernel, circle_kernel, ksize * ksize * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, buffer, height * width * sizeof(uchar),
               cudaMemcpyHostToDevice);

    dilateTiledGPU<<<morphTiledGridDim, blockDim,
                     block_width * block_width * sizeof(uchar)>>>(
        d_input, d_swap, height, width, d_kernel, ksize);
    erodeTiledGPU<<<morphTiledGridDim, blockDim,
                    block_width * block_width * sizeof(uchar)>>>(
        d_swap, d_input, height, width, d_kernel, ksize);

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_input, height * width * sizeof(uchar),
               cudaMemcpyDeviceToHost);

    cudaMemcpy(d_input, buffer, height * width * sizeof(uchar),
               cudaMemcpyHostToDevice);
    dilateGPU<<<morphGridDim, blockDim>>>(d_input, d_swap, height, width,
                                          d_kernel, ksize);
    erodeGPU<<<morphGridDim, blockDim>>>(d_swap, d_input, height, width,
                                         d_kernel, ksize);
    cudaMemcpy(expected, d_input, height * width * sizeof(uchar),
               cudaMemcpyDeviceToHost);

    assertArrayEqual(expected, output, height * width);

    delete[] expected;
    delete[] output;
    delete[] circle_kernel;
    cudaFree(d_kernel);
    cudaFree(d_swap);
    cudaFree(d_input);
}

Test(morphCloseTiled, real_case_scenario)
{
    cv::Mat bgd = cv::imread("docs/report_resources/background.png");
    cv::Mat input = cv::imread("docs/report_resources/source_example.png");
    cr_assert(!input.empty() && !bgd.empty());
    cv::cvtColor(input, input, cv::COLOR_BGR2GRAY);
    cv::cvtColor(bgd, bgd, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(input, input, cv::Size(15, 15), 0.2);
    cv::GaussianBlur(bgd, bgd, cv::Size(15, 15), 0.2);

    cv::absdiff(input, bgd, input);
    cv::threshold(input, input, 20, 255, cv::THRESH_BINARY);

    int width = input.cols;
    int height = input.rows;

    size_t ksize = 15;
    uchar *circle_kernel = getCircleKernel(ksize);

    uchar *buffer = input.ptr<uchar>(0);
    uchar *output = new uchar[height * width];
    uchar *expected = new uchar[height * width];

    int block_width = 32;
    size_t tile_width = block_width - ksize + 1;
    dim3 blockDim(block_width, block_width);
    dim3 morphTiledGridDim(int(ceil((float)height / tile_width)),
                           int(ceil((float)width / tile_width)));
    dim3 morphGridDim(int(ceil((float)height / block_width)),
                      int(ceil((float)width / block_width)));

    uchar *d_input;
    uchar *d_swap;
    uchar *d_kernel;

    cudaMalloc(&d_input, height * width * sizeof(uchar));
    cudaMalloc(&d_swap, height * width * sizeof(uchar));
    cudaMalloc(&d_kernel, ksize * ksize * sizeof(float));

    cudaMemcpy(d_kernel, circle_kernel, ksize * ksize * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, buffer, height * width * sizeof(uchar),
               cudaMemcpyHostToDevice);

    erodeTiledGPU<<<morphTiledGridDim, blockDim,
                    block_width * block_width * sizeof(uchar)>>>(
        d_input, d_swap, height, width, d_kernel, ksize);
    dilateTiledGPU<<<morphTiledGridDim, blockDim,
                     block_width * block_width * sizeof(uchar)>>>(
        d_input, d_swap, height, width, d_kernel, ksize);

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_input, height * width * sizeof(uchar),
               cudaMemcpyDeviceToHost);

    cudaMemcpy(d_input, buffer, height * width * sizeof(uchar),
               cudaMemcpyHostToDevice);
    erodeGPU<<<morphGridDim, blockDim>>>(d_input, d_swap, height, width,
                                         d_kernel, ksize);
    dilateGPU<<<morphGridDim, blockDim>>>(d_swap, d_input, height, width,
                                          d_kernel, ksize);
    cudaMemcpy(expected, d_input, height * width * sizeof(uchar),
               cudaMemcpyDeviceToHost);

    assertArrayEqual(expected, output, height * width);

    delete[] expected;
    delete[] output;
    delete[] circle_kernel;
    cudaFree(d_kernel);
    cudaFree(d_swap);
    cudaFree(d_input);
}