#ifndef __MOD_HPP__
#define __MOD_HPP__

#include <opencv2/opencv.hpp>

#include "utils.hpp"

std::vector<cv::Rect> detectObjectInFrameOpenCV(const cv::Mat &background,
                                                cv::Mat frame);

cv::Mat detectObjectInFrameCPU(const cv::Mat &background, cv::Mat frame);

void grayscale(const cv::Mat &src, s_image dst);

void blur(s_image src, s_image dst, size_t ksize, double sigma);

void diff(s_image src1, s_image scr2, s_image dst);

void morphOpen(s_image src, s_image dst, size_t ksize);

void treshold(s_image src, s_image dst, unsigned char threshold,
              unsigned char maxval);

std::vector<cv::Rect> connectedComponents(s_image src);

#endif