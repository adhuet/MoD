#include <iostream>
#include <opencv2/opencv.hpp>

#include "mod_GPU.hpp"

int runGPU(cv::VideoCapture capture)
{
    std::cout << "Running GPU code" << std::endl;
    cv::Mat frame;
    cv::Mat background;
    capture >> background;
    if (background.empty())
    {
        std::cerr << "First frame of video (background) is empty!" << std::endl;
        return -1;
    }

    cv::namedWindow("Input/Output", cv::WINDOW_AUTOSIZE);
    for (;;)
    {
        capture >> frame;
        if (frame.empty())
            break;

        cv::Mat output = detectObjectInFrameGPU(background, frame);
        cv::cvtColor(output, output, cv::COLOR_GRAY2BGR);
        cv::putText(output, "Diff", cv::Point(10, 10), cv::FONT_HERSHEY_PLAIN,
                    1, cv::Scalar(0, 0, 255), 2);

        // Create a new image to hold the concatenated images
        cv::Mat concat(output.rows, frame.cols + output.cols, output.type());

        // Copy the first image into the left half of the new image
        frame.copyTo(concat(cv::Rect(0, 0, frame.cols, frame.rows)));

        // Copy the second image into the right half of the new image
        output.copyTo(
            concat(cv::Rect(frame.cols, 0, output.cols, output.rows)));

        // Display the concatenated image

        // cv::imshow("Output", frame);
        cv::imshow("Input/Output", concat);
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

    int retval = runGPU(capture);
    return (retval == -1) ? EXIT_FAILURE : EXIT_SUCCESS;
}