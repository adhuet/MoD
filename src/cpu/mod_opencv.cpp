#include <iostream>

#include "mod.hpp"

std::vector<cv::Rect> detectObjectInFrameOpenCV(const cv::Mat &background,
                                                cv::Mat frame)
{
    cv::Mat bgd;
    cv::Mat image;

    // (2) Convert both background and frame to grayscale
    cv::cvtColor(background, bgd, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frame, image, cv::COLOR_BGR2GRAY);

    // (3) Smooth the two images
    cv::GaussianBlur(image, image, cv::Size(15, 15), 0.2);
    cv::GaussianBlur(bgd, bgd, cv::Size(15, 15), 0.2);

    // (4) Compute the difference
    cv::absdiff(image, bgd, image);

    cv::threshold(image, image, 20, 255, cv::ThresholdTypes::THRESH_BINARY);

    // (5) Morphological opening
    cv::Mat kernel =
        cv::getStructuringElement(cv::MORPH_OPEN, cv::Size(15, 15));
    cv::morphologyEx(image, image, cv::MORPH_OPEN, kernel);

    // (6) Threshold the image and get connected components
    cv::Mat labels;
    cv::connectedComponents(image, labels);

    // (7) Compute and output the bboxes
    cv::Mat dst;
    labels.convertTo(dst, CV_8UC1);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(dst, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> bboxes(contours.size());
    for (size_t i = 0; i < contours.size(); i++)
    {
        bboxes[i] = cv::boundingRect(contours[i]);
    }

    return bboxes;
}