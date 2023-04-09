#ifndef __MOD_HPP__
#define __MOD_HPP__

#include <opencv2/opencv.hpp>

#include "utils.hpp"

std::vector<cv::Rect> detectObjectInFrameOpenCV(const cv::Mat &background,
                                                cv::Mat frame);

cv::Mat detectObjectInFrameCPU(const cv::Mat &background, cv::Mat frame);

void grayscale(const cv::Mat &src, SImage &dst);

void blur(const SImage &src, SImage &dst, size_t ksize, double sigma);

void diff(const SImage &src1, const SImage &src2, SImage &dst);

void morphOpen(const SImage &src, SImage &dst, size_t ksize);
void morphClose(const SImage &src, SImage &dst, size_t ksize);

void treshold(const SImage &src, SImage &dst, unsigned char threshold,
              unsigned char maxval);

std::vector<cv::Rect> connectedComponents(SImage src);

#endif