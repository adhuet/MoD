#include <opencv2/opencv.hpp>

#include "utils.hpp"

s_image toPtr(const cv::Mat &image)
{
    int height = image.rows;
    int width = image.cols;
    s_image image_struct;

    image_struct.data = (uchar *)malloc(height * width * sizeof(uchar));
    memcpy(image_struct.data, image.ptr<uchar>(0),
           height * width * sizeof(uchar));
    image_struct.height = height;
    image_struct.width = width;

    return image_struct;
}

cv::Mat toMat(s_image image)
{
    cv::Mat mat = cv::Mat::zeros(cv::Size(image.width, image.height), CV_8UC1);

    uchar *mat_data = mat.ptr<uchar>(0);
    memcpy(mat_data, image.data, image.height * image.width * sizeof(uchar));

    return mat;
}