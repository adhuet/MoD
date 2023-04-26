#include <criterion/criterion.h>
#include <cuda_runtime.h>
#include <iostream>

#include "mod_GPU.hpp"

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
    // clang-format off
    uchar buffer[13 * 14] = {
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

    __attribute__ ((unused)) int expected[13 * 14] = {
          0,   0,   2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   2,   2,   2,   0,   0,   0,   0,   0,   0,  24,  24,   0,   0,
          2,   2,   2,   2,   2,   0,   0,   0,   0,   0,  24,  24,   0,   0,
          0,   2,   2,   2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 129,   0,   0,   0,   0,   0,   0, 136,   0,   0,   0,
          0,   0, 129, 129, 129,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 129,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0  
    };
    // clang-format on

    uchar *d_input;
    int *d_output;

    int *output = (int *)malloc(13 * 14 * sizeof(int));

    cudaMalloc(&d_input, 13 * 14 * sizeof(uchar));
    cudaMalloc(&d_output, 13 * 14 * sizeof(int));
    cudaMemcpy(d_input, buffer, 13 * 14 * sizeof(uchar),
               cudaMemcpyHostToDevice);

    dim3 blockDim(4, 4);
    dim3 gridDim(int(ceil((float)13 / blockDim.x)),
                 int(ceil((float)14 / blockDim.y)));
    connectedComponentsGPU(d_input, d_output, 13, 14, gridDim, blockDim);

    CUDA_WARN(cudaDeviceSynchronize());

    cudaMemcpy(output, d_output, 13 * 14 * sizeof(int), cudaMemcpyDeviceToHost);

    assertArrayEqual(expected, output, 13 * 14);

    cudaFree(d_input);
    cudaFree(d_output);
    free(output);
}