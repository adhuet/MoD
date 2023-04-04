#ifndef __MOD_HPP__
#define __MOD_HPP__

#include <opencv2/opencv.hpp>

std::vector<cv::Rect> detectObjectInFrameOpenCV(const cv::Mat &background,
                                                cv::Mat frame);

cv::Mat detectObjectInFrameCPU(const cv::Mat &background, cv::Mat frame);

void grayscale(const cv::Mat &src, cv::Mat dst);
void blur(cv::Mat src, cv::Mat dst, size_t ksize, double sigma);
void diff(cv::Mat src1, cv::Mat src2, cv::Mat dst);
void morphOpen(cv::Mat src, cv::Mat dst, size_t ksize);
void treshold(cv::Mat src, cv::Mat dst, unsigned char threshold,
              unsigned char maxval);
std::vector<cv::Rect> connectedComponents(cv::Mat src);

#endif