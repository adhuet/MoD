#include <criterion/criterion.h>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "mod.hpp"
#include "unionfind.hpp"
#include "utils.hpp"

static __attribute__((unused)) void printMatrix(cv::Mat mat)
{
    for (int i = 0; i < mat.rows; i++)
    {
        for (int j = 0; j < mat.cols; j++)
            std::cout << static_cast<unsigned>(mat.at<uchar>(i, j)) << " ";
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

// static __attribute__((unused)) void printMatrix(s_image mat)
// {
//     for (int i = 0; i < mat.height; i++)
//     {
//         for (int j = 0; j < mat.width; j++)
//             std::cout << static_cast<unsigned>(mat.data[i * mat.width + j])
//                       << " ";
//         std::cout << std::endl;
//     }
//     std::cout << std::endl;
// }

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

void dilateBinary1(const SImage &src, SImage &dst, uchar *kernel, size_t ksize);
void erodeBinary1(const SImage &src, SImage &dst, uchar *kernel, size_t ksize);
void dilateBinary255(const SImage &src, SImage &dst, uchar *kernel,
                     size_t ksize);
void erodeBinary255(const SImage &src, SImage &dst, uchar *kernel,
                    size_t ksize);

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
    SImage A(AMat);
    SImage B(AMat);

    float kernel[9] = { 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 };

    filter2D(A, B, kernel, 3);
    assertArrayEqual(A.data, B.data, 81);
    free(buffer1);
}

Test(filter2D, simple)
{
    cv::Mat AMat = cv::Mat::eye(3, 3, CV_8UC1);
    SImage A(AMat);
    SImage B(AMat);

    float kernel[9] = { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 };

    uchar expected[9] = { 2, 0, 0, 0, 3, 0, 0, 0, 2 };

    // std::cout << "AMat" << std::endl;
    // printMatrix(AMat);
    // std::cout << "A" << std::endl << A << std::endl;

    filter2D(A, B, kernel, 3);
    // std::cout << "B after filter:" << std::endl << B << std::endl;

    assertArrayEqual(expected, B.data, 9);
}

Test(getGaussianMatrix, simple)
{
    float *kernel = getGaussianMatrix(5, 0.2);

    float sum = 0.0;
    for (size_t i = 0; i < 25; i++)
        sum += kernel[i];

    cr_assert(fabs(sum - 1.0) < 0.01, "Sum is not equal to 1, got sum = %.3f",
              sum);
    delete[] kernel;
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

    SImage image(11, 11, buffer);

    SImage result(11, 11);

    dilateBinary1(image, result, kernel, 3);

    assertArrayEqual(expected, result.data, 11 * 11);
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

    SImage image(13, 13, buffer);

    SImage result(13, 13);

    erodeBinary1(image, result, kernel, 3);

    assertArrayEqual(expected, result.data, 13 * 13);
}

Test(morphological, dilationCross)
{
    uchar buffer[5 * 5] = { 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1,
                            1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0 };

    SImage image(5, 5, buffer);

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

    SImage result(5, 5);

    dilateBinary1(image, result, kernel, 3);
    assertArrayEqual(expected, result.data, 5 * 5);
    // std::cout << "Result:"
    //           << "result.data[0] = " << static_cast<unsigned>(result.data[0])
    //           << std::endl
    //           << result;
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

    SImage image(13, 13, buffer);

    SImage result(13, 13);

    dilateBinary1(image, result, kernel, 9);

    assertArrayEqual(expected, result.data, 13 * 13);
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

    SImage image(13, 13, buffer);

    SImage result(13, 13);

    erodeBinary1(image, result, kernel, 9);

    assertArrayEqual(expected, result.data, 13 * 13);
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

    SImage image(11, 11, buffer);

    SImage result(11, 11);

    dilateBinary255(image, result, kernel, 3);

    assertArrayEqual(expected, result.data, 11 * 11);
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

    SImage image(13, 13, buffer);

    SImage result(13, 13);

    erodeBinary255(image, result, kernel, 3);

    assertArrayEqual(expected, result.data, 13 * 13);
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

    SImage image(13, 13, buffer);

    SImage dilated(13, 13);
    dilateBinary255(image, dilated, kernel, 5);

    SImage dilatedEroded(13, 13);
    erodeBinary255(dilated, dilatedEroded, kernel, 5);

    SImage resultOpened(13, 13);
    morphOpen(image, resultOpened, 5);

    assertArrayEqual(dilatedEroded.data, resultOpened.data, 13 * 13);

    delete[] kernel;
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

    SImage image(13, 13, buffer);

    uchar *kernel = getCircleKernel(5);

    SImage eroded(13, 13);
    erodeBinary255(image, eroded, kernel, 5);

    SImage erodedDilated(13, 13);
    dilateBinary255(eroded, erodedDilated, kernel, 5);

    SImage resultClosed(13, 13);
    morphClose(image, resultClosed, 5);

    assertArrayEqual(erodedDilated.data, resultClosed.data, 13 * 13);

    delete[] kernel;
}

Test(unionFind, findSimple)
{
    int L[] = { 0, 1, 2, 3, 4, 5, 6 };
    cr_assert_eq(find(L, 0), 0, "Expected find(L, 0)=%d, got %d", 0,
                 find(L, 0));
    cr_assert_eq(find(L, 1), 1, "Expected find(L, 1)=%d, got %d", 1,
                 find(L, 1));
    cr_assert_eq(find(L, 2), 2, "Expected find(L, 2)=%d, got %d", 2,
                 find(L, 2));
    cr_assert_eq(find(L, 3), 3, "Expected find(L, 3)=%d, got %d", 3,
                 find(L, 3));
    cr_assert_eq(find(L, 4), 4, "Expected find(L, 4)=%d, got %d", 4,
                 find(L, 4));
    cr_assert_eq(find(L, 5), 5, "Expected find(L, 5)=%d, got %d", 5,
                 find(L, 5));
    cr_assert_eq(find(L, 6), 6, "Expected find(L, 6)=%d, got %d", 6,
                 find(L, 6));
}

Test(unionFind, findHard)
{
    int L[] = { 0, 1, 1, 0, 2, 3, 3, 0 };
    cr_assert_eq(find(L, 4), 1, "Expected find(L, %d)=%d, got %d", 4, 1,
                 find(L, 4));
    cr_assert_eq(find(L, 2), 1, "Expected find(L, %d)=%d, got %d", 2, 1,
                 find(L, 2));
    cr_assert_eq(find(L, 6), 0, "Expected find(L, %d)=%d, got %d", 6, 0,
                 find(L, 6));
}

std::vector<int> rosenfeldNeighbours(int *L, int width, int x, int y);

Test(getNeighbours, simpleNeighbours)
{
    // clang-format off
    int buffer[5 * 5] = {
         0,  1,  2,  3,  4,
         5,  6,  7,  8,  9,
        10, 11, 12, 13, 14,
        15, 16, 17, 18, 19,
        20, 21, 22, 23, 24
    };
    // clang-format on

    std::vector<int> labels = rosenfeldNeighbours(buffer, 5, 2, 2);
    cr_assert_eq(labels.size(), 4);
    cr_assert(std::find(labels.begin(), labels.end(), 6) != labels.end());
    cr_assert(std::find(labels.begin(), labels.end(), 7) != labels.end());
    cr_assert(std::find(labels.begin(), labels.end(), 8) != labels.end());
    cr_assert(std::find(labels.begin(), labels.end(), 11) != labels.end());
}

Test(getNeighbours, onBorder)
{
    // clang-format off
    int buffer[5 * 5] = {
         0,  1,  2,  3,  4,
         5,  6,  7,  8,  9,
        10, 11, 12, 13, 14,
        15, 16, 17, 18, 19,
        20, 21, 22, 23, 24
    };
    // clang-format on

    std::vector<int> labels = rosenfeldNeighbours(buffer, 5, 0, 2);
    cr_assert_eq(labels.size(), 2);
    cr_assert(std::find(labels.begin(), labels.end(), 5) != labels.end());
    cr_assert(std::find(labels.begin(), labels.end(), 6) != labels.end());
}

Test(getNeighbours, oneBackground)
{
    // clang-format off
    int buffer[5 * 5] = {
         0,  1,  2,  3,  4,
         5,  6,  7,  8,  9,
        10, 11, 12, 13, 14,
        15, 16, 17, 18, 19,
        20, 21, 22, 23, 24
    };
    // clang-format on

    std::vector<int> labels = rosenfeldNeighbours(buffer, 5, 1, 1);
    cr_assert_eq(labels.size(), 3);
    cr_assert(std::find(labels.begin(), labels.end(), 1) != labels.end());
    cr_assert(std::find(labels.begin(), labels.end(), 2) != labels.end());
    cr_assert(std::find(labels.begin(), labels.end(), 5) != labels.end());
}

Test(getNeighbours, borderAndBackground)
{
    // clang-format off
    int buffer[5 * 5] = {
         0,  1,  2,  3,  4,
         5,  6,  7,  8,  9,
        10, 11, 12, 13, 14,
        15, 16, 17, 18, 19,
        20, 21, 22, 23, 24
    };
    // clang-format on

    std::vector<int> labels = rosenfeldNeighbours(buffer, 5, 0, 1);
    cr_assert_eq(labels.size(), 1);
    cr_assert(std::find(labels.begin(), labels.end(), 1) != labels.end());
}

Test(getNeighbours, onTopLeftCorner)
{
    // clang-format off
    int buffer[5 * 5] = {
         0,  1,  2,  3,  4,
         5,  6,  7,  8,  9,
        10, 11, 12, 13, 14,
        15, 16, 17, 18, 19,
        20, 21, 22, 23, 24
    };
    // clang-format on

    std::vector<int> labels = rosenfeldNeighbours(buffer, 5, 0, 0);
    cr_assert_eq(labels.size(), 0);
}

Test(getNeighbours, onBotRightCorner)
{
    // clang-format off
    int buffer[5 * 5] = {
         0,  1,  2,  3,  4,
         5,  6,  7,  8,  9,
        10, 11, 12, 13, 14,
        15, 16, 17, 18, 19,
        20, 21, 22, 23, 24
    };
    // clang-format on

    std::vector<int> labels = rosenfeldNeighbours(buffer, 5, 4, 4);
    cr_assert_eq(labels.size(), 3);
    cr_assert(std::find(labels.begin(), labels.end(), 18) != labels.end());
    cr_assert(std::find(labels.begin(), labels.end(), 19) != labels.end());
    cr_assert(std::find(labels.begin(), labels.end(), 23) != labels.end());
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

    __attribute__ ((unused)) uchar expected[13 * 14] = {
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

    SImage image(14, 13, buffer);
    cv::Mat mat = image.toCVMat();

    cv::Mat tmp;
    tmp.create(mat.size(), CV_16U);

    // cv::CCL_DEFAULT
    // cv::CCL_WU
    // cv::CCL_GRANA
    // cv::CCL_BOLELLI

    // std::cout << "Input:" << std::endl;
    // std::cout << image << std::endl;
    cv::connectedComponents(mat, tmp, 8, cv::CCL_BOLELLI);
    int *labelled = (int *)malloc(image.size * sizeof(int));
    connectedComponents(image, labelled);
    SImage otp(image.width, image.height, labelled);
    free(labelled);
    // std::cout << "Output:" << std::endl;
    // std::cout << otp << std::endl;

    assertArrayEqual(expected, otp.data, otp.size);
    /*
        cv::Mat output = otp.toCVMat();
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(output, contours, cv::RETR_EXTERNAL,
                         cv::CHAIN_APPROX_SIMPLE);

        std::vector<cv::Rect> bboxes(contours.size());
        for (size_t i = 0; i < contours.size(); i++)
        {
            bboxes[i] = cv::boundingRect(contours[i]);
        }
        // for (int r = 0; r < output.rows; r++)
        //     for (int c = 0; c < output.cols; c++)
        //         output.at<uchar>(r, c) = static_cast<uchar>(tmp.at<int>(r,
       c));

        std::cout << "Result of connected components: " << std::endl;
        std::cout << SImage(output) << std::endl;

        std::cout << "Resulting bounding boxes:" << std::endl;
        for (auto bbox : bboxes)
            std::cout << bbox << std::endl;
    */
}

Test(connectedComponents, convexComp, .timeout = 3)
{
    // clang-format off
    __attribute__ ((unused)) uchar buffer[13 * 14] = {
          0, 255,   0,   0,   0,   0, 255, 255, 255, 255,   0,   0,   0,   0,
        255, 255,   0,   0,   0,   0, 255, 255, 255, 255,   0,   0,   0,   0,
        255, 255,   0,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,
        255, 255,   0,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,
          0, 255,   0,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,
          0, 255, 255,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,
          0, 255, 255, 255,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,
          0,   0, 255, 255,   0,   0, 255, 255, 255,   0,   0,   0,   0,   0,
          0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
    };

    __attribute__ ((unused)) uchar expected[13 * 14] = {
          0,   1,   0,   0,   0,   0,   1,   1,   1,   1,   0,   0,   0,   0,
          1,   1,   0,   0,   0,   0,   1,   1,   1,   1,   0,   0,   0,   0,
          1,   1,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,
          1,   1,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,
          0,   1,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,
          0,   1,   1,   0,   0,   0,   0,   0,   1,   1,   0,   0,   0,   0,
          0,   1,   1,   1,   0,   0,   0,   1,   1,   0,   0,   0,   0,   0,
          0,   0,   1,   1,   0,   0,   1,   1,   1,   0,   0,   0,   0,   0,
          0,   0,   1,   1,   1,   1,   1,   1,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   1,   1,   1,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
    };
    // clang-format on

    SImage image(14, 13, buffer);
    cv::Mat mat = image.toCVMat();

    cv::Mat tmp;
    tmp.create(mat.size(), CV_16U);

    // std::cout << "Input:" << std::endl;
    // std::cout << image << std::endl;
    cv::connectedComponents(mat, tmp, 8, cv::CCL_BOLELLI);
    int *labelled = (int *)malloc(image.size * sizeof(int));
    connectedComponents(image, labelled);
    SImage otp(image.width, image.height, labelled);
    free(labelled);
    // std::cout << "Output:" << std::endl;
    // std::cout << otp << std::endl;

    assertArrayEqual(expected, otp.data, otp.size);
}

/*
// FIXME top left corner is always background due to raster index being 0
Test(connectedComponents, topLeftCornerBeingAPain, .timeout = 3)
{
    // clang-format off
    __attribute__ ((unused)) uchar buffer[13 * 14] = {
        255, 255,   0,   0,   0,   0, 255, 255, 255, 255,   0,   0,   0,   0,
        255, 255,   0,   0,   0,   0, 255, 255, 255, 255,   0,   0,   0,   0,
        255, 255,   0,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,
        255, 255,   0,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,
          0, 255,   0,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,
          0, 255, 255,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,
          0, 255, 255, 255,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,
          0,   0, 255, 255,   0,   0, 255, 255, 255,   0,   0,   0,   0,   0,
          0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
    };

    __attribute__ ((unused)) uchar expected[13 * 14] = {
          1,   1,   0,   0,   0,   0,   1,   1,   1,   1,   0,   0,   0,   0,
          1,   1,   0,   0,   0,   0,   1,   1,   1,   1,   0,   0,   0,   0,
          1,   1,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,
          1,   1,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,
          0,   1,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,
          0,   1,   1,   0,   0,   0,   0,   0,   1,   1,   0,   0,   0,   0,
          0,   1,   1,   1,   0,   0,   0,   1,   1,   0,   0,   0,   0,   0,
          0,   0,   1,   1,   0,   0,   1,   1,   1,   0,   0,   0,   0,   0,
          0,   0,   1,   1,   1,   1,   1,   1,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   1,   1,   1,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
    };
    // clang-format on

    SImage image(14, 13, buffer);
    cv::Mat mat = image.toCVMat();

    cv::Mat tmp;
    tmp.create(mat.size(), CV_16U);

    // cv::CCL_DEFAULT
    // cv::CCL_WU
    // cv::CCL_GRANA
    // cv::CCL_BOLELLI

    std::cout << "Input:" << std::endl;
    std::cout << image << std::endl;
    cv::connectedComponents(mat, tmp, 8, cv::CCL_BOLELLI);
    int *labelled = (int *)malloc(image.size * sizeof(int));
    connectedComponents(image, labelled);
    SImage otp(image.width, image.height, labelled);
    free(labelled);
    std::cout << "Output:" << std::endl;
    std::cout << otp << std::endl;

    assertArrayEqual(expected, otp.data, otp.size);
}*/

Test(bboxes, noObject)
{
    // clang-format off
    int buffer[5 * 5] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0
    };
    // clang-format on

    std::vector<cv::Rect> bboxes = getBoundingBoxes(buffer, 5, 5);
    cr_assert(bboxes.size() == 0, "Expected 0, got %d", bboxes.size());
}

Test(bboxes, simple)
{
    // clang-format off
    int buffer[5 * 5] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0
    };
    // clang-format on

    std::vector<cv::Rect> bboxes = getBoundingBoxes(buffer, 5, 5);
    cr_assert_eq(bboxes.size(), 1, "Expected bboxes.size() = 1, got %d",
                 bboxes.size());
    cv::Rect box = bboxes[0];

    cr_assert(box.x == 2, "Expected box.x = 2, got %d", box.x);
    cr_assert(box.y == 2, "Expected box.y = 2, got %d", box.y);
    cr_assert(box.width == 1, "Expected box.width = 1, got %d", box.width);
    cr_assert(box.height == 1, "Expected box.height = 1, got %d", box.height);
}

Test(bboxes, largeBox)
{
    // clang-format off
    int buffer[5 * 5] = {
        0, 0, 0, 0, 0,
        0, 1, 1, 1, 0,
        0, 1, 1, 1, 0,
        0, 1, 1, 1, 0,
        0, 0, 0, 0, 0
    };
    // clang-format on

    std::vector<cv::Rect> bboxes = getBoundingBoxes(buffer, 5, 5);
    cr_assert_eq(bboxes.size(), 1, "Expected bboxes.size() = 1, got %d",
                 bboxes.size());
    cv::Rect box = bboxes[0];

    cr_assert(box.x == 1, "Expected box.x = 1, got %d", box.x);
    cr_assert(box.y == 1, "Expected box.y = 1, got %d", box.y);
    cr_assert(box.width == 3, "Expected box.width = 3, got %d", box.width);
    cr_assert(box.height == 3, "Expected box.height = 3, got %d", box.height);
}

Test(bboxes, biggerBox)
{
    // clang-format off
    int buffer[5 * 5] = {
        0, 0, 0, 0, 0,
        0, 1, 1, 0, 0,
        0, 1, 1, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0
    };
    // clang-format on

    std::vector<cv::Rect> bboxes = getBoundingBoxes(buffer, 5, 5);
    cr_assert_eq(bboxes.size(), 1, "Expected bboxes.size() = 1, got %d",
                 bboxes.size());
    cv::Rect box = bboxes[0];

    cr_assert(box.x == 1, "Expected box.x = 1, got %d", box.x);
    cr_assert(box.y == 1, "Expected box.y = 1, got %d", box.y);
    cr_assert(box.width == 2, "Expected box.width = 2, got %d", box.width);
    cr_assert(box.height == 2, "Expected box.height = 2, got %d", box.height);
}

Test(bboxes, twoBoxes)
{
    // clang-format off
    int buffer[5 * 5] = {
        1, 1, 0, 0, 0,
        1, 1, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 2, 2, 0,
        0, 0, 0, 0, 0
    };
    // clang-format on

    std::vector<cv::Rect> bboxes = getBoundingBoxes(buffer, 5, 5);
    cr_assert_eq(bboxes.size(), 2, "Expected bboxes.size() = 2, got %d",
                 bboxes.size());
    cv::Rect box1 = bboxes[0];
    cv::Rect box2 = bboxes[1];

    cr_assert(box2.x == 0, "Expected box2.x = 0, got %d", box2.x);
    cr_assert(box2.y == 0, "Expected box2.y = 0, got %d", box2.y);
    cr_assert(box2.width == 2, "Expected box2.width = 2, got %d", box2.width);
    cr_assert(box2.height == 2, "Expected box2.height = 2, got %d",
              box2.height);

    cr_assert(box1.x == 2, "Expected box1.x = 2, got %d", box1.x);
    cr_assert(box1.y == 3, "Expected box1.y = 3, got %d", box1.y);
    cr_assert(box1.width == 2, "Expected box1.width = 2, got %d", box1.width);
    cr_assert(box1.height == 1, "Expected box1.height = 1, got %d",
              box1.height);
}