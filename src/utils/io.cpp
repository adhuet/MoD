#include <opencv2/opencv.hpp>

#include "utils.hpp"

SImage::SImage(int width, int height)
    : width(width)
    , height(height)
    , size(width * height)
    , nb_bytes(width * height * sizeof(uchar))
{
    this->data = (uchar *)malloc(this->nb_bytes);
}

SImage::SImage(int width, int height, uchar *buffer)
    : width(width)
    , height(height)
    , size(width * height)
    , nb_bytes(width * height * sizeof(uchar))
{
    this->data = (uchar *)malloc(this->nb_bytes);
    memcpy(data, buffer, this->nb_bytes);
}

SImage::SImage(const cv::Mat &mat)
    : width(mat.cols)
    , height(mat.rows)
    , size(width * height)
    , nb_bytes(width * height * sizeof(uchar))
{
    data = (uchar *)malloc(this->nb_bytes);
    memcpy(data, mat.ptr<uchar>(0), this->nb_bytes);
}

SImage::~SImage()
{
    free(data);
}

SImage::SImage(const SImage &other)
{
    if (this != &other)
    {
        this->width = other.width;
        this->height = other.height;
        this->size = other.size;
        this->nb_bytes = other.nb_bytes;
        this->data = (uchar *)malloc(this->nb_bytes);
        memcpy(this->data, other.data, this->nb_bytes);
    }
}

cv::Mat SImage::toCVMat()
{
    cv::Mat mat = cv::Mat::zeros(cv::Size(this->width, this->height), CV_8UC1);
    uchar *mat_data = mat.ptr<uchar>(0);
    memcpy(mat_data, this->data, this->nb_bytes);

    return mat;
}

std::ostream &operator<<(std::ostream &os, const SImage &img)
{
    for (int i = 0; i < img.height; i++)
    {
        for (int j = 0; j < img.width; j++)
            os << static_cast<unsigned>(img.data[i * img.width + j]) << " ";
        os << std::endl;
    }
    os << std::endl;
    return os;
}