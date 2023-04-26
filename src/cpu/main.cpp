#include <iostream>
#include <opencv2/opencv.hpp>

#include "mod.hpp"
#include "utils.hpp"

int runOpenCV(cv::VideoCapture capture)
{
    cv::Mat frame;
    cv::Mat background;
    capture >> background;
    if (background.empty())
    {
        std::cerr << "First frame of video (background) is empty!" << std::endl;
        return -1;
    }

    cv::namedWindow("OpenCV", cv::WINDOW_AUTOSIZE);
    for (;;)
    {
        capture >> frame;
        if (frame.empty())
            break;

        std::vector<cv::Rect> bboxes =
            detectObjectInFrameOpenCV(background, frame);

        cv::Mat output;
        frame.copyTo(output);

        for (const auto &bbox : bboxes)
            cv::rectangle(output, bbox, cv::Scalar(0, 0, 255), 2);

        // Create a new image to hold the concatenated images
        cv::Mat concat(output.rows, frame.cols + output.cols, output.type());

        // Copy the first image into the left half of the new image
        frame.copyTo(concat(cv::Rect(0, 0, frame.cols, frame.rows)));

        // Copy the second image into the right half of the new image
        output.copyTo(
            concat(cv::Rect(frame.cols, 0, output.cols, output.rows)));

        // Display the concatenated image
        cv::imshow("OpenCV", concat);
        if (cv::waitKey(20) >= 0)
            break;
    }
    return 0;
}

int runCPU(cv::VideoCapture capture)
{
    cv::Mat frame;
    cv::Mat background;
    capture >> background;
    if (background.empty())
    {
        std::cerr << "First frame of video (background) is empty!" << std::endl;
        return -1;
    }

    int framenb = 0;
    cv::namedWindow("CPU", cv::WINDOW_AUTOSIZE);
    for (;;)
    {
        framenb++;
        capture >> frame;
        if (frame.empty())
            break;

        auto bboxes = detectObjectInFrameCPU(background, frame);
        // cv::Mat output = detectObjectInFrameCPU(background, frame);

        cv::Mat output;
        frame.copyTo(output);
        for (const auto &bbox : bboxes)
            cv::rectangle(output, bbox, cv::Scalar(0, 0, 255), 2);

        cv::putText(output, "Detected", cv::Point(10, 10),
                    cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 2);

        // Create a new image to hold the concatenated images
        cv::Mat concat(output.rows, frame.cols + output.cols, output.type());

        // Copy the first image into the left half of the new image
        frame.copyTo(concat(cv::Rect(0, 0, frame.cols, frame.rows)));

        // Copy the second image into the right half of the new image
        output.copyTo(
            concat(cv::Rect(frame.cols, 0, output.cols, output.rows)));

        // Display the concatenated image
        cv::imshow("CPU", concat);
        if (cv::waitKey(20) >= 0)
            break;
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << argv[0] << "Usage: " << argv[0] << " [VIDEO_FILENAME]"
                  << std::endl;
        return EXIT_FAILURE;
    }

    std::string filename(argv[1]);
    cv::VideoCapture capture(filename);

    if (!capture.isOpened())
    {
        std::cerr << "Opening video failed!" << std::endl;
        return EXIT_FAILURE;
    }

    int retval = runCPU(capture);
    return (retval == -1) ? EXIT_FAILURE : EXIT_SUCCESS;
}