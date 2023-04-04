#include "display.hpp"

void displayVideo(cv::VideoCapture video)
{
    cv::Mat frame;
    cv::namedWindow("Input", cv::WindowFlags::WINDOW_AUTOSIZE);
    for (;;)
    {
        video >> frame;
        if (frame.empty())
            break;
        cv::imshow("Input", frame);
        if (cv::waitKey(20) >= 0) // Stop at any point by pressing any key
            break;
    }
}