#include <criterion/criterion.h>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>

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

static void assertArrayEqual(uchar *arr1, uchar *arr2, int n)
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