#include "mod.hpp"

void grayscale(const cv::Mat &src, cv::Mat dst)
{
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            cv::Vec3b bgr = src.at<cv::Vec3b>(i, j);
            uchar blue = bgr[0];
            uchar green = bgr[1];
            uchar red = bgr[2];

            uchar gray = 0.2989 * red + 0.5870 * green + 0.1140 * blue;

            dst.at<uchar>(i, j) = gray;
        }
    }
}