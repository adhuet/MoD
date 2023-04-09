#include <criterion/criterion.h>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "mod.hpp"
#include "utils.hpp"

static __attribute__((unused)) void printMatrix(cv::Mat mat)
{
    for (int i = 0; i < mat.rows; i++)
    {
        for (int j = 0; j < mat.cols; j++)
            std::cout << static_cast<unsigned>(mat.at<uchar>(j, i)) << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

static __attribute__((unused)) void printMatrix(float *mat, int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
            std::cout << std::setprecision(1) << mat[i * size + j] << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

static __attribute__((unused)) void printMatrix(s_image mat)
{
    for (int i = 0; i < mat.height; i++)
    {
        for (int j = 0; j < mat.width; j++)
            std::cout << static_cast<unsigned>(mat.data[i * mat.width + j])
                      << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

static __attribute__((unused)) void printMatrix(uchar *mat, int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
            std::cout << static_cast<unsigned>(mat[i * size + j]) << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

static void assertArrayEqual(uchar *arr1, uchar *arr2, int n)
{
    for (int i = 0; i < n; i++)
        cr_assert_eq(arr1[i], arr2[i],
                     "Expected arr1[%d] = %d, got arr2[%d] = %d", i, arr1[i], i,
                     arr2[i]);
}

void dilateBinary1(s_image src, s_image dst, uchar *kernel, size_t ksize);
void erodeBinary1(s_image src, s_image dst, uchar *kernel, size_t ksize);
void dilateBinary255(s_image src, s_image dst, uchar *kernel, size_t ksize);
void erodeBinary255(s_image src, s_image dst, uchar *kernel, size_t ksize);

Test(check, pass)
{
    cr_assert(1);
}

Test(filter2D, identity)
{
    uchar *buffer1 = (uchar *)malloc(81 * sizeof(uchar));
    for (size_t i = 0; i < 81; i++)
    {
        buffer1[i] = static_cast<uchar>(rand() % 256);
    }

    cv::Mat AMat(9, 9, CV_8UC1, buffer1);
    s_image A = toPtr(AMat);
    s_image B = toPtr(AMat);

    float kernel[9] = { 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 };

    filter2D(A, B, kernel, 3);
    assertArrayEqual(A.data, B.data, 81);
    free(buffer1);
    free(A.data);
    free(B.data);
}

Test(filter2D, simple)
{
    cv::Mat AMat = cv::Mat::eye(3, 3, CV_8UC1);
    s_image A = toPtr(AMat);
    s_image B = toPtr(AMat);

    float kernel[9] = { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 };

    uchar expected[9] = { 2, 0, 0, 0, 3, 0, 0, 0, 2 };

    // std::cout << "A" << std::endl;
    // printMatrix(A);

    filter2D(A, B, kernel, 3);
    // std::cout << "B after filter:" << std::endl;
    // printMatrix(B);

    assertArrayEqual(B.data, expected, 9);
    free(A.data);
    free(B.data);
}

Test(getGaussianMatrix, simple)
{
    float *kernel = getGaussianMatrix(5, 0.2);

    float sum = 0.0;
    for (size_t i = 0; i < 25; i++)
        sum += kernel[i];

    cr_assert(fabs(sum - 1.0) < 0.01, "Sum is not equal to 1, got sum = %.3f",
              sum);
}

Test(morphological, dilationWikipedia)
{
    // clang-format off
    uchar buffer[11 * 11] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0,
        0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0,
        0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    uchar kernel[3 * 3] = {
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
    };

    uchar expected[11 * 11] = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0
    };
    // clang-format on

    s_image image;
    image.height = 11;
    image.width = 11;
    image.data = buffer;

    s_image result = { 11, 11, (uchar *)malloc(11 * 11 * sizeof(uchar)) };

    dilateBinary1(image, result, kernel, 3);

    assertArrayEqual(expected, result.data, 11 * 11);

    free(result.data);
}

Test(morphological, erosionWikipedia)
{
    // clang-format off
    uchar buffer[13 * 13] = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    };

    uchar kernel[3 * 3] = {
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
    };


    uchar expected[13 * 13] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    // clang-format on

    s_image image;
    image.height = 13;
    image.width = 13;
    image.data = buffer;

    s_image result = { 13, 13, (uchar *)malloc(13 * 13 * sizeof(uchar)) };

    erodeBinary1(image, result, kernel, 3);

    assertArrayEqual(expected, result.data, 13 * 13);

    free(result.data);
}

Test(morphological, dilationCross)
{
    uchar buffer[5 * 5] = { 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1,
                            1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0 };

    s_image image;
    image.height = 5;
    image.width = 5;
    image.data = buffer;

    uchar kernel[3 * 3] = { 0, 1, 0, 1, 1, 1, 0, 1, 0 };

    // clang-format off
    uchar expected[5 * 5] = {
        0, 1, 1, 1, 0,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        0, 1, 1, 1, 0
    };
    // clang-format on

    s_image result = { 5, 5, (uchar *)malloc(5 * 5 * sizeof(uchar)) };

    dilateBinary1(image, result, kernel, 3);
    assertArrayEqual(expected, result.data, 5 * 5);
    // std::cout << "Result:"
    //           << "result.data[0] = " << static_cast<unsigned>(result.data[0])
    //           << std::endl;
    // printMatrix(result);

    free(result.data);
}

Test(morphological, dilateCircle)
{
    // clang-format off
    uchar buffer[13 * 13] = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    };

    uchar kernel[9 * 9] = {
        0, 0, 1, 1, 1, 1, 1, 0, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 1, 1, 1, 1, 1, 1, 1, 0,
        0, 0, 1, 1, 1, 1, 1, 0, 0
    };

    uchar expected[13 * 13] = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    };
    // clang-format on

    s_image image;
    image.height = 13;
    image.width = 13;
    image.data = buffer;

    s_image result = { 13, 13, (uchar *)malloc(13 * 13 * sizeof(uchar)) };

    dilateBinary1(image, result, kernel, 9);

    assertArrayEqual(expected, result.data, 13 * 13);

    free(result.data);
}

Test(morphological, erodeCircle)
{
    // clang-format off
    uchar buffer[13 * 13] = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    };

    uchar kernel[9 * 9] = {
        0, 0, 1, 1, 1, 1, 1, 0, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 1, 1, 1, 1, 1, 1, 1, 0,
        0, 0, 1, 1, 1, 1, 1, 0, 0
    };

    uchar expected[13 * 13] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    // clang-format on

    s_image image;
    image.height = 13;
    image.width = 13;
    image.data = buffer;

    s_image result = { 13, 13, (uchar *)malloc(13 * 13 * sizeof(uchar)) };

    erodeBinary1(image, result, kernel, 9);

    assertArrayEqual(expected, result.data, 13 * 13);

    free(result.data);
}

Test(morphological, dilation255)
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

    uchar expected[255 * 255] = {
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

    s_image image;
    image.height = 11;
    image.width = 11;
    image.data = buffer;

    s_image result = { 11, 11, (uchar *)malloc(11 * 11 * sizeof(uchar)) };

    dilateBinary255(image, result, kernel, 3);

    assertArrayEqual(expected, result.data, 11 * 11);

    free(result.data);
}

Test(morphological, erosion255)
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
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
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

    s_image image;
    image.height = 13;
    image.width = 13;
    image.data = buffer;

    s_image result = { 13, 13, (uchar *)malloc(13 * 13 * sizeof(uchar)) };

    erodeBinary255(image, result, kernel, 3);

    assertArrayEqual(expected, result.data, 13 * 13);

    free(result.data);
}

Test(morphObject, circleSimple)
{
    // clang-format off
    uchar expected[9 * 9] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 1, 0, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 0,
        0, 0, 1, 1, 1, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    // clang-format on

    uchar *result = getCircleKernel(9);

    // s_image res = { 9, 9, result };

    assertArrayEqual(expected, result, 9 * 9);
    delete[] result;
}

Test(morphological, morphOpen_1)
{
    // clang-format off

    /* Kernel
        0 0 0 0 0
        0 1 1 1 0
        0 1 1 1 0
        0 1 1 1 0
        0 0 0 0 0
    */
    uchar buffer[13 * 13] = {
          0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0, 255, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0,
        255, 255, 255, 255, 255,   0,   0,   0,   0,   0, 255, 255,   0,
          0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0, 255,   0,   0,
          0,   0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
    };

    __attribute__ ((unused)) uchar expectedDilate[13 * 13] = {
        255, 255, 255, 255, 255,   0,   0,   0,   0, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255,   0,   0,   0, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255,   0,   0,   0, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255,   0,   0,   0, 255, 255, 255, 255,
        255, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,
          0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0, 255, 255, 255,   0,   0,   0,   0, 255, 255, 255,   0,
          0, 255, 255, 255, 255, 255,   0,   0,   0, 255, 255, 255,   0,
          0, 255, 255, 255, 255, 255,   0,   0,   0, 255, 255, 255,   0,
          0, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,
          0,   0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0
    };

    __attribute__ ((unused)) uchar expected[13 * 13] = {
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0, 255, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0,
          0, 255, 255, 255, 255,   0,   0,   0,   0,   0, 255, 255,   0,
          0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0, 255,   0,   0,
          0,   0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
    };
    // clang-format on

    uchar *kernel = getCircleKernel(5);

    s_image image = { 13, 13, buffer };

    s_image dilated = { 13, 13, (uchar *)malloc(13 * 13 * sizeof(uchar)) };

    dilateBinary255(image, dilated, kernel, 5);

    s_image dilatedEroded = { 13, 13,
                              (uchar *)malloc(13 * 13 * sizeof(uchar)) };

    erodeBinary255(dilated, dilatedEroded, kernel, 5);

    s_image resultOpened = { 13, 13, (uchar *)malloc(13 * 13 * sizeof(uchar)) };

    morphOpen(image, resultOpened, 5);

    assertArrayEqual(dilatedEroded.data, resultOpened.data, 13 * 13);

    delete[] kernel;
    free(dilated.data);
    free(dilatedEroded.data);
    free(resultOpened.data);
}

Test(morphological, morphClose_1)
{
    // clang-format off
    uchar buffer[13 * 13] = {
          0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0, 255, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255,   0,
        255, 255, 255, 255, 255,   0,   0,   0,   0,   0, 255, 255,   0,
          0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0, 255,   0,   0,
          0,   0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
    };
    // clang-format on

    s_image image = { 13, 13, buffer };
    uchar *kernel = getCircleKernel(5);

    s_image eroded = { 13, 13, (uchar *)malloc(13 * 13 * sizeof(uchar)) };
    erodeBinary255(image, eroded, kernel, 5);

    s_image erodedDilated = { 13, 13,
                              (uchar *)malloc(13 * 13 * sizeof(uchar)) };
    dilateBinary255(eroded, erodedDilated, kernel, 5);

    s_image resultClosed = { 13, 13, (uchar *)malloc(13 * 13 * sizeof(uchar)) };
    morphClose(image, resultClosed, 5);

    assertArrayEqual(erodedDilated.data, resultClosed.data, 13 * 13);

    free(kernel);
    free(eroded.data);
    free(erodedDilated.data);
    free(resultClosed.data);
}