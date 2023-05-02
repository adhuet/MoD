#include <iostream>
#include <opencv2/opencv.hpp>

#include "mod_GPU.hpp"
#include "utils.hpp"

static __attribute__((unused)) void printMatrix(int *mat, int height, int width)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
            std::cout << std::setfill(' ') << std::setw(6) << mat[i * width + j]
                      << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

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

    // Some cuda events for simple fps timing
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0.0f;

    cv::namedWindow("GPU", cv::WINDOW_AUTOSIZE);
    size_t totalFrame = 0;
    for (;;)
    {
        std::cout << '\r' << "Frame: " << totalFrame << std::flush;
        capture >> frame;
        if (frame.empty())
            break;

        // Simple fps timer
        cudaEventRecord(start, 0);

        std::vector<cv::Rect> bboxes =
            detectObjectInFrameGPU(background, frame);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        float fps = 1.0f / (milliseconds / 1000.0f);
        std::ostringstream ostream;
        ostream << "Fps: " << std::fixed << std::setprecision(2) << fps;
        std::string fps_string = ostream.str();

        cv::Mat output;
        frame.copyTo(output);

        for (const auto &bbox : bboxes)
            cv::rectangle(output, bbox, cv::Scalar(0, 0, 255), 2);
        cv::putText(output, fps_string, cv::Point(10, 10),
                    cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 2);

        // Create a new image to hold the concatenated images
        cv::Mat concat(output.rows, frame.cols + output.cols, output.type());

        // Copy the first image into the left half of the new image
        frame.copyTo(concat(cv::Rect(0, 0, frame.cols, frame.rows)));

        // Copy the second image into the right half of the new image
        output.copyTo(
            concat(cv::Rect(frame.cols, 0, output.cols, output.rows)));

        totalFrame++;

        // Display the concatenated image
        cv::imshow("GPU", concat);
        if (cv::waitKey(20) >= 0)
            break;
    }
    std::cout << std::endl;

    cudaEventDestroy(stop);
    cudaEventDestroy(start);

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

    // int retval = runGPU(capture);
    int retval = renderObjectsInCaptureGPU(capture);
    return (retval == -1) ? EXIT_FAILURE : EXIT_SUCCESS;
}