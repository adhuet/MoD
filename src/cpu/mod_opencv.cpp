#include "mod.hpp"

#include <iostream>

std::vector<cv::Rect> detectObjectInFrameOpenCV(const cv::Mat &background,
                                                cv::Mat frame)
{
    cv::Mat bgd;
    cv::Mat image;

    // (2) Convert both background and frame to grayscale
    cv::cvtColor(background, bgd, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frame, image, cv::COLOR_BGR2GRAY);

    // (3) Smooth the two images
    cv::GaussianBlur(image, image, cv::Size(5, 5), 0.2);
    cv::GaussianBlur(bgd, bgd, cv::Size(5, 5), 0.2);

    // (4) Compute the difference
    cv::absdiff(image, bgd, image);

    // (5) Morphological opening
    cv::Mat kernel =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
    cv::morphologyEx(image, image, cv::MORPH_OPEN, kernel);

    // (6) Threshold the image and get connected components
    cv::threshold(image, image, 20, 255, cv::ThresholdTypes::THRESH_BINARY);
    cv::imshow("Before connected components", image);
    cv::waitKey(20);
    std::cout << "Doing connected components..." << std::endl;

    cv::Mat labels;
    int nLabels = cv::connectedComponents(image, labels);
    (void)nLabels;

    // std::vector<cv::Vec3b> colors(nLabels);
    // colors[0] = cv::Vec3b(0, 0, 0); // background
    // for (int label = 1; label < nLabels; ++label)
    // {
    //     colors[label] =
    //         cv::Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
    // }
    //
    // cv::Mat dst(image.size(), CV_8UC3);
    // for (int r = 0; r < dst.rows; ++r)
    // {
    //     for (int c = 0; c < dst.cols; ++c)
    //     {
    //         int label = labels.at<int>(r, c);
    //         cv::Vec3b &pixel = dst.at<cv::Vec3b>(r, c);
    //         pixel = colors[label];
    //     }
    // }

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