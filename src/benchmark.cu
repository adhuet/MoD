// #include <cuda_runtime.h>
#include <chrono>
#include <cuda_runtime.h>
#include <exception>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "mod.hpp"
#include "mod_GPU.hpp"
#include "utils.hpp"

#ifndef _CPU_VERSION
#    define _CPU_VERSION 0.0
#endif

#ifndef _GPU_VERSION
#    define _GPU_VERSION 0.0
#endif

#ifndef _OCV_VERSION
#    define _OCV_VERSION 0.0
#endif

struct BM_times
{
    double grayscale;
    double blur;
    double diff;
    double threshold;
    double morph;
    double kernel_erodeGPU;
    double kernel_dilateGPU;
    double connectedComps;
    double kernel_initCCL;
    double kernel_mergeCCL;
    double kernel_compressCCL;
    double bboxes;
    double other;
    double get_gaussian_matrix;
    double get_circle_kernel;
    double gpu_mem_management;
};

typedef struct BM_times BM_times;

void erodeBinary255(const SImage &src, SImage &dst, uchar *kernel,
                    size_t ksize);
void dilateBinary255(const SImage &src, SImage &dst, uchar *kernel,
                     size_t ksize);

BM_times benchGPU(cv::VideoCapture capture, dim3 gridDim, dim3 blockDim)
{
    BM_times timers = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    cv::Mat frame;
    cv::Mat background;
    capture >> background;
    if (background.empty())
        throw std::runtime_error("First frame of video (background is empty!)");

    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t start_step;
    cudaEventCreate(&start_step);
    cudaEvent_t end_step;
    cudaEventCreate(&end_step);
    cudaEvent_t end;
    cudaEventCreate(&end);
    cudaEvent_t start_substep;
    cudaEventCreate(&start_substep);
    cudaEvent_t end_substep;
    cudaEventCreate(&end_substep);

    float total_duration = 0.0f;
    float duration = 0.0f;
    float sub_duration = 0.0f;

    // Initialize all important constants
    const int height = background.rows;
    const int width = background.cols;
    const int numPixels = height * width;
    const size_t ksize = 15;

    cudaEventRecord(start_step, 0);
    const float *gaussianKernel = getGaussianMatrix(ksize, 2.0);
    cudaEventRecord(end_step, 0);
    cudaEventSynchronize(end_step);
    cudaEventElapsedTime(&duration, start_step, end_step);
    timers.get_gaussian_matrix += duration;

    const uchar threshold = 20;
    const uchar maxval_tresh = 255;
    const int morphological_circle_diameter = 15;

    cudaEventRecord(start_step, 0);
    const uchar *circleKernel = getCircleKernel(morphological_circle_diameter);
    cudaEventRecord(end_step, 0);
    cudaEventSynchronize(end_step);
    cudaEventElapsedTime(&duration, start_step, end_step);
    timers.get_circle_kernel += duration;

    // Host buffers used during computation
    cudaEventRecord(start_step, 0);
    int *labels = new int[numPixels]; // Holds the final CCL symbollic image
    cudaEventRecord(end_step, 0);
    cudaEventSynchronize(end_step);
    cudaEventElapsedTime(&duration, start_step, end_step);
    timers.gpu_mem_management += duration;

    std::vector<cv::Rect> bboxes; // Holds the bboxes for a specific frame

    // Device buffers used during computation
    uchar3 *d_background; // This holds the original background temporarily
    uchar3 *d_frame; // This holds the original frame temporarily

    uchar *d_bgd; // This holds the gray blurred background throughout the whole
                  // capture processing
    uchar *d_input; // This hold the frame throughout all of the processing
    uchar *d_swap; // This allows copy and manipulation of the frame

    float *d_gaussianKernel; // Device buffer for the blur kernel
    uchar *d_circleKernel; // Device buffer for the morphological kernel

    int *d_labels; // This holds the symbollic image after CCL

    cudaEventRecord(start_step, 0);
    // Allocations necessary for background
    cudaMalloc(&d_bgd, numPixels * sizeof(uchar));
    cudaMalloc(&d_gaussianKernel, ksize * ksize * sizeof(float));
    cudaMalloc(&d_circleKernel,
               morphological_circle_diameter * morphological_circle_diameter
                   * sizeof(uchar));
    cudaMalloc(&d_background, numPixels * sizeof(uchar3));
    // Initialization
    // FIXME, put those into constant memory
    cudaMemcpy(d_gaussianKernel, gaussianKernel, ksize * ksize * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_circleKernel, circleKernel,
               morphological_circle_diameter * morphological_circle_diameter
                   * sizeof(uchar),
               cudaMemcpyHostToDevice);

    // Background processing
    // Copy background to device
    cudaMemcpy(d_background, background.ptr<uchar3>(0),
               numPixels * sizeof(uchar3), cudaMemcpyHostToDevice);
    cudaEventRecord(end_step, 0);
    cudaEventSynchronize(end_step);
    cudaEventElapsedTime(&duration, start_step, end_step);
    timers.gpu_mem_management += duration;

    // Process background
    cudaEventRecord(start_step, 0);
    grayscaleGPU<<<gridDim, blockDim>>>(d_background, d_bgd, height, width);
    cudaEventRecord(end_step, 0);
    cudaEventSynchronize(end_step);
    cudaEventElapsedTime(&duration, start_step, end_step);
    timers.grayscale += duration;

    cudaEventRecord(start_step, 0);
    blurGPU<<<gridDim, blockDim>>>(d_bgd, d_bgd, height, width,
                                   d_gaussianKernel, ksize);
    cudaEventRecord(end_step, 0);
    cudaEventSynchronize(end_step);
    cudaEventElapsedTime(&duration, start_step, end_step);
    timers.blur += duration;

    // We do not this the original background buffer, so we release memory as
    // soon as possible
    cudaEventRecord(start_step, 0);
    cudaFree(d_background);
    // Rest of the allocations for the frame processing
    cudaMalloc(&d_frame, numPixels * sizeof(uchar3));
    cudaMalloc(&d_input, numPixels * sizeof(uchar));
    cudaMalloc(&d_swap, numPixels * sizeof(uchar));
    cudaMalloc(&d_labels, numPixels * sizeof(int));
    cudaEventRecord(end_step, 0);
    cudaEventSynchronize(end_step);
    cudaEventElapsedTime(&duration, start_step, end_step);
    timers.gpu_mem_management += duration;

    for (;;)
    {
        cudaEventRecord(start, 0);
        capture >> frame;
        if (frame.empty())
            break;

        cudaEventRecord(start_step, 0);
        cudaMemcpy(d_frame, frame.ptr<uchar3>(0), numPixels * sizeof(uchar3),
                   cudaMemcpyHostToDevice);
        cudaEventRecord(end_step, 0);
        cudaEventSynchronize(end_step);
        cudaEventElapsedTime(&duration, start_step, end_step);
        timers.gpu_mem_management += duration;

        // grayscale
        cudaEventRecord(start_step, 0);
        grayscaleGPU<<<gridDim, blockDim>>>(d_frame, d_input, height, width);
        cudaEventRecord(end_step, 0);
        cudaEventSynchronize(end_step);
        cudaEventElapsedTime(&duration, start_step, end_step);
        timers.grayscale += duration;

        // blur
        cudaEventRecord(start_step, 0);
        blurGPU<<<gridDim, blockDim>>>(d_input, d_input, height, width,
                                       d_gaussianKernel, ksize);
        cudaEventRecord(end_step, 0);
        cudaEventSynchronize(end_step);
        cudaEventElapsedTime(&duration, start_step, end_step);
        timers.blur += duration;

        // diff
        cudaEventRecord(start_step, 0);
        diffGPU<<<gridDim, blockDim>>>(d_bgd, d_input, d_input, height, width);
        cudaEventRecord(end_step, 0);
        cudaEventSynchronize(end_step);
        cudaEventElapsedTime(&duration, start_step, end_step);
        timers.diff += duration;

        // threshold
        cudaEventRecord(start_step, 0);
        thresholdGPU<<<gridDim, blockDim>>>(d_input, d_input, height, width,
                                            threshold, maxval_tresh);
        cudaEventRecord(end_step, 0);
        cudaEventSynchronize(end_step);
        cudaEventElapsedTime(&duration, start_step, end_step);
        timers.threshold += duration;

        // morph
        cudaEventRecord(start_step, 0);

        cudaEventRecord(start_substep, 0);
        dilateGPU<<<gridDim, blockDim>>>(d_input, d_swap, height, width,
                                         d_circleKernel, ksize);
        cudaEventRecord(end_substep, 0);
        cudaEventSynchronize(end_substep);
        cudaEventElapsedTime(&sub_duration, start_substep, end_substep);
        timers.kernel_dilateGPU += sub_duration;

        cudaEventRecord(start_substep, 0);
        erodeGPU<<<gridDim, blockDim>>>(d_swap, d_input, height, width,
                                        d_circleKernel, ksize);
        cudaEventRecord(end_substep, 0);
        cudaEventSynchronize(end_substep);
        cudaEventElapsedTime(&sub_duration, start_substep, end_substep);
        timers.kernel_erodeGPU += sub_duration;

        cudaEventRecord(end_step, 0);

        cudaEventSynchronize(end_step);
        cudaEventElapsedTime(&duration, start_step, end_step);
        timers.morph += duration;

        // connectedComps
        cudaEventRecord(start_step, 0);

        cudaEventRecord(start_substep, 0);
        initCCL<<<gridDim, blockDim>>>(d_input, d_labels, height, width);
        cudaEventRecord(end_substep, 0);
        cudaEventSynchronize(end_substep);
        cudaEventElapsedTime(&sub_duration, start_substep, end_substep);
        timers.kernel_initCCL += sub_duration;

        cudaEventRecord(start_substep, 0);
        mergeCCL<<<gridDim, blockDim>>>(d_input, d_labels, height, width);
        cudaEventRecord(end_substep, 0);
        cudaEventSynchronize(end_substep);
        cudaEventElapsedTime(&sub_duration, start_substep, end_substep);
        timers.kernel_mergeCCL += sub_duration;

        cudaEventRecord(start_substep, 0);
        compressCCL<<<gridDim, blockDim>>>(d_input, d_labels, height, width);
        cudaEventRecord(end_substep, 0);
        cudaEventSynchronize(end_substep);
        cudaEventElapsedTime(&sub_duration, start_substep, end_substep);
        timers.kernel_compressCCL += sub_duration;

        cudaEventRecord(start_substep, 0);
        cudaMemcpy(labels, d_labels, height * width * sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaEventRecord(end_substep, 0);
        cudaEventSynchronize(end_substep);
        cudaEventElapsedTime(&sub_duration, start_substep, end_substep);
        timers.gpu_mem_management += sub_duration;

        cudaEventRecord(end_step, 0);
        cudaEventSynchronize(end_step);
        cudaEventElapsedTime(&duration, start_step, end_step);
        timers.connectedComps += duration;

        // bboxes
        cudaEventRecord(start_step, 0);
        std::vector<cv::Rect> bboxes = getBoundingBoxes(labels, width, height);
        cudaEventRecord(end_step, 0);
        cudaEventSynchronize(end_step);
        cudaEventElapsedTime(&duration, start_step, end_step);
        timers.bboxes += duration;

        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&duration, start, end);
        total_duration += duration;
    }
    std::cout << "FPS: " << std::setprecision(4)
              << capture.get(cv::CAP_PROP_FRAME_COUNT)
            / (total_duration / 1000.0f)
              << std::endl;

    cudaEventRecord(start_step, 0);
    cudaFree(d_labels);
    cudaFree(d_swap);
    cudaFree(d_input);
    cudaFree(d_frame);
    cudaFree(d_circleKernel);
    cudaFree(d_gaussianKernel);
    cudaFree(d_bgd);
    delete[] labels;
    delete[] circleKernel;
    delete[] gaussianKernel;
    cudaEventRecord(end_step, 0);
    cudaEventSynchronize(end_step);
    cudaEventElapsedTime(&duration, start_step, end_step);
    timers.gpu_mem_management += duration;

    cudaEventDestroy(start);
    cudaEventDestroy(start_step);
    cudaEventDestroy(end_step);
    cudaEventDestroy(end);
    cudaEventDestroy(start_substep);
    cudaEventDestroy(end_substep);

    return timers;
}

BM_times benchCPU(cv::VideoCapture capture)
{
    BM_times timers = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    cv::Mat frame;
    cv::Mat background;
    capture >> background;
    if (background.empty())
        throw std::runtime_error("First frame of video (background is empty!)");

    // uchar *kernel;
    // int width = cv::CAP_PROP_FRAME_WIDTH;
    // int height = cv::CAP_PROP_FRAME_HEIGHT;
    auto start = std::chrono::high_resolution_clock::now();
    auto start_step = start;
    auto end_step = start;
    auto end = start;
    auto start_substep = start;
    auto end_substep = start;

    float total_duration = 0.0f;

    for (;;)
    {
        start = std::chrono::high_resolution_clock::now();
        capture >> frame;
        if (frame.empty())
            break;

        SImage bgd(background);
        SImage image(frame);

        // grayscale
        start_step = std::chrono::high_resolution_clock::now();
        grayscale(background, image);
        grayscale(frame, image);
        end_step = std::chrono::high_resolution_clock::now();
        timers.grayscale += static_cast<double>(
            std::chrono::duration_cast<std::chrono::milliseconds>(end_step
                                                                  - start_step)
                .count());

        // blur
        // bgd
        // blur(bgd, bgd, 15, 0.2);
        start_step = std::chrono::high_resolution_clock::now();
        start_substep = start_step;
        float *gaussianMatrix = getGaussianMatrix(15, 0.2);
        end_substep = std::chrono::high_resolution_clock::now();
        filter2D(bgd, bgd, gaussianMatrix, 15);
        delete[] gaussianMatrix;
        end_step = std::chrono::high_resolution_clock::now();
        timers.get_gaussian_matrix += static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                end_substep - start_substep)
                .count());
        timers.blur += static_cast<double>(
            std::chrono::duration_cast<std::chrono::milliseconds>(end_step
                                                                  - start_step)
                .count());
        // image
        // blur(image, image, 15, 0.2);
        start_step = std::chrono::high_resolution_clock::now();
        start_substep = start_step;
        gaussianMatrix = getGaussianMatrix(15, 0.2);
        end_substep = std::chrono::high_resolution_clock::now();
        filter2D(image, image, gaussianMatrix, 15);
        delete[] gaussianMatrix;
        end_step = std::chrono::high_resolution_clock::now();
        timers.get_gaussian_matrix += static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                end_substep - start_substep)
                .count());
        timers.blur += static_cast<double>(
            std::chrono::duration_cast<std::chrono::milliseconds>(end_step
                                                                  - start_step)
                .count());

        // diff
        start_step = std::chrono::high_resolution_clock::now();
        diff(bgd, image, image);
        end_step = std::chrono::high_resolution_clock::now();
        timers.diff += static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(end_step
                                                                  - start_step)
                .count());

        // threshold
        start_step = std::chrono::high_resolution_clock::now();
        threshold(image, image, 20, 255);
        end_step = std::chrono::high_resolution_clock::now();
        timers.threshold += static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(end_step
                                                                  - start_step)
                .count());

        // morph
        start_step = std::chrono::high_resolution_clock::now();
        // morphOpen(image, image, 15);
        start_substep = std::chrono::high_resolution_clock::now();
        uchar *kernel = getCircleKernel(15);
        end_substep = std::chrono::high_resolution_clock::now();
        SImage tmp = image;
        dilateBinary255(image, tmp, kernel, 15);
        erodeBinary255(tmp, image, kernel, 15);
        end_step = std::chrono::high_resolution_clock::now();
        timers.get_circle_kernel += static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                end_substep - start_substep)
                .count());
        timers.morph += static_cast<double>(
            std::chrono::duration_cast<std::chrono::milliseconds>(end_step
                                                                  - start_step)
                .count());
        // connectedComps
        start_step = std::chrono::high_resolution_clock::now();
        int *labels = (int *)malloc(image.width * image.height * sizeof(int));
        connectedComponents(image, labels);
        end_step = std::chrono::high_resolution_clock::now();
        timers.connectedComps += static_cast<double>(
            std::chrono::duration_cast<std::chrono::milliseconds>(end_step
                                                                  - start_step)
                .count());
        // bboxes
        start_step = std::chrono::high_resolution_clock::now();
        std::vector<cv::Rect> bboxes =
            getBoundingBoxes(labels, image.width, image.height);
        free(labels);
        end_step = std::chrono::high_resolution_clock::now();
        timers.bboxes += static_cast<double>(
            std::chrono::duration_cast<std::chrono::milliseconds>(end_step
                                                                  - start_step)
                .count());
        end = std::chrono::high_resolution_clock::now();
        total_duration += static_cast<double>(
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count());
    }
    std::cout << "FPS: " << std::setprecision(4)
              << capture.get(cv::CAP_PROP_FRAME_COUNT)
            / (total_duration / 1000.0f)
              << std::endl;
    return timers;
}

BM_times benchOpenCV(cv::VideoCapture capture)
{
    BM_times timers = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    cv::Mat frame;
    cv::Mat background;
    capture >> background;
    if (background.empty())
        throw std::runtime_error("First frame of video (background is empty!)");

    cv::Mat bgd;
    cv::Mat image;
    cv::Mat kernel;
    cv::Mat labels;
    cv::Mat dst;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Rect> bboxes;

    auto start = std::chrono::high_resolution_clock::now();
    auto start_step = start;
    auto end_step = start;
    auto end = start;

    float total_duration = 0.0f;

    for (;;)
    {
        start = std::chrono::high_resolution_clock::now();
        capture >> frame;
        if (frame.empty())
            break;

        // grayscale
        start_step = std::chrono::high_resolution_clock::now();
        cv::cvtColor(background, bgd, cv::COLOR_BGR2GRAY);
        cv::cvtColor(frame, image, cv::COLOR_BGR2GRAY);
        end_step = std::chrono::high_resolution_clock::now();
        timers.grayscale += static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(end_step
                                                                  - start_step)
                .count());

        // blur
        start_step = std::chrono::high_resolution_clock::now();
        cv::GaussianBlur(image, image, cv::Size(15, 15), 0.2);
        cv::GaussianBlur(bgd, bgd, cv::Size(15, 15), 0.2);
        end_step = std::chrono::high_resolution_clock::now();
        timers.blur += static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(end_step
                                                                  - start_step)
                .count());

        // diff
        start_step = std::chrono::high_resolution_clock::now();
        cv::absdiff(image, bgd, image);
        end_step = std::chrono::high_resolution_clock::now();
        timers.diff += static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(end_step
                                                                  - start_step)
                .count());

        // threshold
        start_step = std::chrono::high_resolution_clock::now();
        cv::threshold(image, image, 20, 255, cv::ThresholdTypes::THRESH_BINARY);
        end_step = std::chrono::high_resolution_clock::now();
        timers.threshold += static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(end_step
                                                                  - start_step)
                .count());
        // morph
        start_step = std::chrono::high_resolution_clock::now();
        kernel = cv::getStructuringElement(cv::MORPH_OPEN, cv::Size(15, 15));
        cv::morphologyEx(image, image, cv::MORPH_OPEN, kernel);
        end_step = std::chrono::high_resolution_clock::now();
        timers.morph += static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(end_step
                                                                  - start_step)
                .count());
        // connectedComps
        start_step = std::chrono::high_resolution_clock::now();
        cv::connectedComponents(image, labels);
        end_step = std::chrono::high_resolution_clock::now();
        timers.connectedComps += static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(end_step
                                                                  - start_step)
                .count());
        // bboxes
        start_step = std::chrono::high_resolution_clock::now();
        labels.convertTo(dst, CV_8UC1);
        cv::findContours(dst, contours, cv::RETR_EXTERNAL,
                         cv::CHAIN_APPROX_SIMPLE);

        bboxes = std::vector<cv::Rect>(contours.size());
        for (size_t i = 0; i < contours.size(); i++)
        {
            bboxes[i] = cv::boundingRect(contours[i]);
        }
        end_step = std::chrono::high_resolution_clock::now();
        timers.bboxes += static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(end_step
                                                                  - start_step)
                .count());
        end = std::chrono::high_resolution_clock::now();
        total_duration += static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count());
    }
    std::cout << "FPS: " << std::setprecision(4)
              << capture.get(cv::CAP_PROP_FRAME_COUNT)
            / (total_duration / 1000000.0f)
              << std::endl;
    return timers;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << argv[0] << " Usage: " << argv[0] << " [VIDEO_FILENAME]"
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

    double nb_frames = capture.get(cv::CAP_PROP_FRAME_COUNT);
    int width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);

    // float test_duration = 0.0; // 15000.01;
    // float test_duration2 = 0.0; // 0.01;
    // float test_percent = 0.0;

    std::cout << std::setfill('#') << std::setw(75) << "\n";
    std::cout << "Filename: " << argv[1] << std::endl;
    std::cout << "Dimensions: " << width << "x" << height << std::endl;
    std::cout << "Framerate: " << capture.get(cv::CAP_PROP_FPS) << "fps"
              << std::endl;
    std::cout << "Total nb of frames: " << nb_frames << std::endl;
    std::cout << "Duration: "
              << 1000 * capture.get(cv::CAP_PROP_FRAME_COUNT)
            / capture.get(cv::CAP_PROP_FPS)
              << "ms" << std::endl;

    std::cout << std::setfill('-') << std::setw(75) << "\n";
    std::cout << "OPENCV Bench (v" << _OCV_VERSION << "):" << std::endl;

    // Run opencv bench
    auto start = std::chrono::high_resolution_clock::now();
    BM_times timers = benchOpenCV(capture);
    auto end = std::chrono::high_resolution_clock::now();
    double duration = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count());
    std::cout << std::endl;
    std::cout << "STEP" << std::setfill(' ')
              << std::setw(24 - std::string("STEP").size()) << "|"
              << std::setw(15) << "FRAME_AVG" << std::setw(17) << "TOTAL"
              << std::setw(17) << "EXEC_TIME" << std::endl;
    std::cout << std::endl;
    std::cout << "grayscale" << std::setfill(' ')
              << std::setw(24 - std::string("grayscale").size()) << "|"
              << std::setw(13) << std::setprecision(7)
              << timers.grayscale / nb_frames << "μs" << std::setw(15)
              << timers.grayscale << "μs" << std::setw(16)
              << std::setprecision(3) << timers.grayscale / duration * 100
              << "%" << std::endl;
    std::cout << "blur" << std::setfill(' ')
              << std::setw(24 - std::string("blur").size()) << "|"
              << std::setw(13) << std::setprecision(7)
              << timers.blur / nb_frames << "μs" << std::setw(15) << timers.blur
              << "μs" << std::setw(16) << std::setprecision(3)
              << timers.blur / duration * 100 << "%" << std::endl;
    std::cout << "diff" << std::setfill(' ')
              << std::setw(24 - std::string("diff").size()) << "|"
              << std::setw(13) << std::setprecision(7)
              << timers.diff / nb_frames << "μs" << std::setw(15) << timers.diff
              << "μs" << std::setw(16) << std::setprecision(3)
              << timers.diff / duration * 100 << "%" << std::endl;
    std::cout << "threshold" << std::setfill(' ')
              << std::setw(24 - std::string("threshold").size()) << "|"
              << std::setw(13) << std::setprecision(7)
              << timers.threshold / nb_frames << "μs" << std::setw(15)
              << timers.threshold << "μs" << std::setw(16)
              << std::setprecision(3) << timers.threshold / duration * 100
              << "%" << std::endl;
    std::cout << "morph" << std::setfill(' ')
              << std::setw(24 - std::string("morph").size()) << "|"
              << std::setw(13) << std::setprecision(7)
              << timers.morph / nb_frames << "μs" << std::setw(15)
              << timers.morph << "μs" << std::setw(16) << std::setprecision(3)
              << timers.morph / duration * 100 << "%" << std::endl;
    std::cout << "connectedComps" << std::setfill(' ')
              << std::setw(24 - std::string("connectedComps").size()) << "|"
              << std::setw(13) << std::setprecision(7)
              << timers.connectedComps / nb_frames << "μs" << std::setw(15)
              << timers.connectedComps << "μs" << std::setw(16)
              << std::setprecision(3) << timers.connectedComps / duration * 100
              << "%" << std::endl;
    std::cout << "bboxes" << std::setfill(' ')
              << std::setw(24 - std::string("bboxes").size()) << "|"
              << std::setw(13) << std::setprecision(7)
              << timers.bboxes / nb_frames << "μs" << std::setw(15)
              << timers.bboxes << "μs" << std::setw(16) << std::setprecision(3)
              << timers.bboxes / duration * 100 << "%" << std::endl;

    std::cout << std::endl
              << "Start to finish: " << duration / 1000.0f << "ms" << std::endl;

    /*
    capture.set(cv::CAP_PROP_POS_FRAMES, 0);
    std::cout << std::setfill('-') << std::setw(75) << "\n";
    std::cout << "CPU Bench (v" << _CPU_VERSION << "):" << std::endl;

    // Run cpu bench
    start = std::chrono::high_resolution_clock::now();
    timers = benchCPU(capture);
    end = std::chrono::high_resolution_clock::now();
    duration = static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count());

    std::cout << std::endl;
    std::cout << "STEP" << std::setfill(' ')
              << std::setw(24 - std::string("STEP").size()) << "|"
              << std::setw(15) << "FRAME_AVG" << std::setw(17) << "TOTAL"
              << std::setw(17) << "EXEC_TIME" << std::endl;
    std::cout << std::endl;
    std::cout << "grayscale" << std::setfill(' ')
              << std::setw(24 - std::string("grayscale").size()) << "|"
              << std::setw(13) << std::setprecision(7)
              << timers.grayscale / nb_frames << "ms" << std::setw(15)
              << timers.grayscale << "ms" << std::setw(16)
              << std::setprecision(3) << timers.grayscale / duration * 100
              << "%" << std::endl;
    std::cout << "blur" << std::setfill(' ')
              << std::setw(24 - std::string("blur").size()) << "|"
              << std::setw(13) << std::setprecision(7)
              << timers.blur / nb_frames << "ms" << std::setw(15) << timers.blur
              << "ms" << std::setw(16) << std::setprecision(3)
              << timers.blur / (duration * 1000.0f) * 100 << "%" << std::endl;
    std::cout << "  - getGaussianMatrix" << std::setfill(' ')
              << std::setw(24 - std::string("  - getGaussianMatrix").size())
              << "|" << std::setw(13) << std::setprecision(7)
              << timers.get_gaussian_matrix / nb_frames << "μs" << std::setw(15)
              << timers.get_gaussian_matrix << "μs" << std::setw(16)
              << std::setprecision(3)
              << timers.get_gaussian_matrix / duration * 100 << "%"
              << std::endl;
    std::cout << "diff" << std::setfill(' ')
              << std::setw(24 - std::string("diff").size()) << "|"
              << std::setw(13) << std::setprecision(7)
              << timers.diff / nb_frames << "μs" << std::setw(15) << timers.diff
              << "μs" << std::setw(16) << std::setprecision(3)
              << (timers.diff / (duration * 1000.0f)) * 100 << "%" << std::endl;
    std::cout << "threshold" << std::setfill(' ')
              << std::setw(24 - std::string("threshold").size()) << "|"
              << std::setw(13) << std::setprecision(7)
              << timers.threshold / nb_frames << "μs" << std::setw(15)
              << timers.threshold << "μs" << std::setw(16)
              << std::setprecision(3)
              << timers.threshold / (duration * 1000.0f) * 100 << "%"
              << std::endl;
    std::cout << "morph" << std::setfill(' ')
              << std::setw(24 - std::string("morph").size()) << "|"
              << std::setw(13) << std::setprecision(7)
              << timers.morph / nb_frames << "ms" << std::setw(15)
              << timers.morph << "ms" << std::setw(16) << std::setprecision(3)
              << timers.morph / duration * 100 << "%" << std::endl;
    std::cout << "  - getCircleKernel" << std::setfill(' ')
              << std::setw(24 - std::string("  - getCircleKernel").size())
              << "|" << std::setw(13) << std::setprecision(7)
              << timers.get_circle_kernel / nb_frames << "μs" << std::setw(15)
              << timers.get_circle_kernel << "μs" << std::setw(16)
              << std::setprecision(3)
              << timers.get_circle_kernel / (duration * 1000.0f) * 100 << "%"
              << std::endl;
    std::cout << "connectedComps" << std::setfill(' ')
              << std::setw(24 - std::string("connectedComps").size()) << "|"
              << std::setw(13) << std::setprecision(7)
              << timers.connectedComps / nb_frames << "ms" << std::setw(15)
              << timers.connectedComps << "ms" << std::setw(16)
              << std::setprecision(3) << timers.connectedComps / duration * 100
              << "%" << std::endl;
    std::cout << "bboxes" << std::setfill(' ')
              << std::setw(24 - std::string("bboxes").size()) << "|"
              << std::setw(13) << std::setprecision(7)
              << timers.bboxes / nb_frames << "ms" << std::setw(15)
              << timers.bboxes << "ms" << std::setw(16) << std::setprecision(3)
              << timers.bboxes / duration * 100 << "%" << std::endl;
    // std::cout << "Other" << std::setfill(' ')
    //           << std::setw(24 - std::string("Other").size()) << "|"
    //           << std::setw(13) << std::setprecision(7) << test_duration <<
    //           "ms"
    //           << std::setw(15) << test_duration2 << "ms" << std::setw(16)
    //           << std::setprecision(3) << test_percent << "%" << std::endl;

    std::cout << std::endl
              << "Start to finish: " << duration / 1000.0f << "s" << std::endl;
    */

    capture.set(cv::CAP_PROP_POS_FRAMES, 0);
    std::cout << std::setfill('-') << std::setw(75) << "\n";
    std::cout << "GPU Bench (v" << _GPU_VERSION << "):" << std::endl;

    dim3 blockDim(64, 64);
    dim3 gridDim(int(ceil((float)width / blockDim.x)),
                 int(ceil((float)height / blockDim.y)));

    std::cout << "blockDim: " << blockDim.x << "x" << blockDim.y << std::endl;
    std::cout << "gridDim: " << gridDim.x << "x" << gridDim.y << std::endl;

    // Run gpu bench
    cudaEvent_t start_bench;
    cudaEvent_t stop_bench;
    cudaEventCreate(&start_bench);
    cudaEventCreate(&stop_bench);
    float gpu_duration = 0.0f;

    cudaEventRecord(start_bench, 0);
    timers = benchGPU(capture, blockDim, gridDim);
    cudaEventRecord(stop_bench, 0);
    cudaEventSynchronize(stop_bench);
    cudaEventElapsedTime(&gpu_duration, start_bench, stop_bench);

    std::cout << std::endl;
    std::cout << "STEP" << std::setfill(' ')
              << std::setw(24 - std::string("STEP").size()) << "|"
              << std::setw(15) << "FRAME_AVG" << std::setw(17) << "TOTAL"
              << std::setw(17) << "EXEC_TIME" << std::endl;
    std::cout << std::endl;
    std::cout << "grayscaleGPU" << std::setfill(' ')
              << std::setw(24 - std::string("grayscaleGPU").size()) << "|"
              << std::setw(13) << std::setprecision(7)
              << timers.grayscale / nb_frames << "ms" << std::setw(15)
              << timers.grayscale << "ms" << std::setw(16)
              << std::setprecision(3) << timers.grayscale / gpu_duration * 100
              << "%" << std::endl;
    std::cout << "blurGPU" << std::setfill(' ')
              << std::setw(24 - std::string("blurGPU").size()) << "|"
              << std::setw(13) << std::setprecision(7)
              << timers.blur / nb_frames << "ms" << std::setw(15) << timers.blur
              << "ms" << std::setw(16) << std::setprecision(3)
              << timers.blur / gpu_duration * 100 << "%" << std::endl;
    std::cout << "  - getGaussianMatrix" << std::setfill(' ')
              << std::setw(24 - std::string("  - getGaussianMatrix").size())
              << "|" << std::setw(13) << std::setprecision(7)
              << timers.get_gaussian_matrix / nb_frames << "ms" << std::setw(15)
              << timers.get_gaussian_matrix << "ms" << std::setw(16)
              << std::setprecision(3)
              << timers.get_gaussian_matrix / gpu_duration * 100 << "%"
              << std::endl;
    std::cout << "diffGPU" << std::setfill(' ')
              << std::setw(24 - std::string("diffGPU").size()) << "|"
              << std::setw(13) << std::setprecision(7)
              << timers.diff / nb_frames << "ms" << std::setw(15) << timers.diff
              << "ms" << std::setw(16) << std::setprecision(3)
              << timers.diff / gpu_duration * 100 << "%" << std::endl;
    std::cout << "thresholdGPU" << std::setfill(' ')
              << std::setw(24 - std::string("thresholdGPU").size()) << "|"
              << std::setw(13) << std::setprecision(7)
              << timers.threshold / nb_frames << "ms" << std::setw(15)
              << timers.threshold << "ms" << std::setw(16)
              << std::setprecision(3) << timers.threshold / gpu_duration * 100
              << "%" << std::endl;
    std::cout << "morph" << std::setfill(' ')
              << std::setw(24 - std::string("morph").size()) << "|"
              << std::setw(13) << std::setprecision(7)
              << timers.morph / nb_frames << "ms" << std::setw(15)
              << timers.morph << "ms" << std::setw(16) << std::setprecision(3)
              << timers.morph / gpu_duration * 100 << "%" << std::endl;
    std::cout << "  - dilateGPU" << std::setfill(' ')
              << std::setw(24 - std::string("  - dilateGPU").size()) << "|"
              << std::setw(13) << std::setprecision(7)
              << timers.kernel_dilateGPU / nb_frames << "ms" << std::setw(15)
              << timers.kernel_dilateGPU << "ms" << std::setw(16)
              << std::setprecision(3)
              << timers.kernel_dilateGPU / gpu_duration * 100 << "%"
              << std::endl;
    std::cout << "  - erodeGPU" << std::setfill(' ')
              << std::setw(24 - std::string("  - erodeGPU").size()) << "|"
              << std::setw(13) << std::setprecision(7)
              << timers.kernel_erodeGPU / nb_frames << "ms" << std::setw(15)
              << timers.kernel_erodeGPU << "ms" << std::setw(16)
              << std::setprecision(3)
              << timers.kernel_erodeGPU / gpu_duration * 100 << "%"
              << std::endl;
    std::cout << "  - getCircleKernel" << std::setfill(' ')
              << std::setw(24 - std::string("  - getCircleKernel").size())
              << "|" << std::setw(13) << std::setprecision(7)
              << timers.get_circle_kernel / nb_frames << "ms" << std::setw(15)
              << timers.get_circle_kernel << "ms" << std::setw(16)
              << std::setprecision(3)
              << timers.get_circle_kernel / gpu_duration * 100 << "%"
              << std::endl;
    std::cout << "connectedComps" << std::setfill(' ')
              << std::setw(24 - std::string("connectedComps").size()) << "|"
              << std::setw(13) << std::setprecision(7)
              << timers.connectedComps / nb_frames << "ms" << std::setw(15)
              << timers.connectedComps << "ms" << std::setw(16)
              << std::setprecision(3)
              << timers.connectedComps / gpu_duration * 100 << "%" << std::endl;
    std::cout << "  - initCCL" << std::setfill(' ')
              << std::setw(24 - std::string("  - initCCL").size()) << "|"
              << std::setw(13) << std::setprecision(7)
              << timers.kernel_initCCL / nb_frames << "ms" << std::setw(15)
              << timers.kernel_initCCL << "ms" << std::setw(16)
              << std::setprecision(3)
              << timers.kernel_initCCL / gpu_duration * 100 << "%" << std::endl;
    std::cout << "  - mergeCCL" << std::setfill(' ')
              << std::setw(24 - std::string("  - mergeCCL").size()) << "|"
              << std::setw(13) << std::setprecision(7)
              << timers.kernel_mergeCCL / nb_frames << "ms" << std::setw(15)
              << timers.kernel_mergeCCL << "ms" << std::setw(16)
              << std::setprecision(3)
              << timers.kernel_mergeCCL / gpu_duration * 100 << "%"
              << std::endl;
    std::cout << "  - compressCCL" << std::setfill(' ')
              << std::setw(24 - std::string("  - compressCCL").size()) << "|"
              << std::setw(13) << std::setprecision(7)
              << timers.kernel_compressCCL / nb_frames << "ms" << std::setw(15)
              << timers.kernel_compressCCL << "ms" << std::setw(16)
              << std::setprecision(3)
              << timers.kernel_compressCCL / gpu_duration * 100 << "%"
              << std::endl;
    std::cout << "bboxes" << std::setfill(' ')
              << std::setw(24 - std::string("bboxes").size()) << "|"
              << std::setw(13) << std::setprecision(7)
              << timers.bboxes / nb_frames << "ms" << std::setw(15)
              << timers.bboxes << "ms" << std::setw(16) << std::setprecision(3)
              << timers.bboxes / gpu_duration * 100 << "%" << std::endl;
    // std::cout << "Other" << std::setfill(' ')
    //           << std::setw(24 - std::string("Other").size()) << "|"
    //           << std::setw(13) << std::setprecision(7) << test_duration <<
    //           "ms"
    //           << std::setw(15) << test_duration2 << "ms" << std::setw(16)
    //           << std::setprecision(3) << test_percent << "%" << std::endl;
    std::cout << "Mem. Management" << std::setfill(' ')
              << std::setw(24 - std::string("Mem. Management").size()) << "|"
              << std::setw(13) << std::setprecision(7)
              << timers.gpu_mem_management / nb_frames << "ms" << std::setw(15)
              << timers.gpu_mem_management << "ms" << std::setw(16)
              << std::setprecision(3)
              << timers.gpu_mem_management / gpu_duration * 100 << "%"
              << std::endl;

    std::cout << std::endl
              << "Start to finish: " << gpu_duration / 1000.0f << "s"
              << std::endl;
    /*
    ##################################################
    Filename: truc.mp4
    Dimensions: 500x500
    Framerate: 30fps
    Total nb of frames: 125
    Length: 35000ms
    --------------------------------------------------
    OPENCV Bench (vOCV_VERSION):
    FPS: 25.0                                                // (updating)
    STEP                    |    FRAME-AVG          TOTAL    // (after run)

    grayscale               |   00000.00ms         0.00ms
    blur                    |       0.00ms         0.00ms
    diff                    |       0.00ms         0.00ms
    threshold               |       0.00ms         0.00ms
    morph                   |       0.00ms         0.00ms
    connectedComps          |       0.00ms         0.00ms
    bboxes                  |       0.00ms         0.00ms

    Start to finish: 0000    - getCircleKernel   .00ms // (bold)

    --------------------------------------------------
    CPU Bench (vCPU_VERSION):
    FPS: 25.0                                                // (updating)
    STEP                    |    FRAME-AVG          TOTAL    // (after run)

    grayscale               |   00000.00ms         0.00ms
    blur                    |       0.00ms         0.00ms
    diff                    |       0.00ms         0.00ms
    threshold               |       0.00ms         0.00ms
    morph                   |       0.00ms         0.00ms
    connectedComps          |       0.00ms         0.00ms
    bboxes                  |       0.00ms         0.00ms
    Other                   |       0.00ms         0.00ms
        - getGaussianMatrix |       0.00ms         0.00ms
        - getCircleKernel   |       0.00ms         0.00ms

    Start to finish: 0000.00ms                               // (bold)

    --------------------------------------------------
    GPU Bench (vGPU_VERSION):
    blockDim: (32x32)
    gridDim: (12x12)
    FPS: 25.0                                                // (updating)
    STEP                    |    FRAME-AVG          TOTAL    // (after run)

    grayscaleGPU            |   00000.00ms        0.00ms
    blurGPU                 |       0.00ms        0.00ms
    diffGPU                 |       0.00ms        0.00ms
    thresholdGPU            |       0.00ms        0.00ms
    dilateGPU               |       0.00ms        0.00ms
    erodeGPU                |       0.00ms        0.00ms
    connectedComps          |       0.00ms        0.00ms
        - initCCL           |       0.00ms        0.00ms
        - mergeCCL          |       0.00ms        0.00ms
        - compressCCL       |       0.00ms        0.00ms
    bboxes                  |       0.00ms        0.00ms
    Other                   |       0.00ms        0.00ms
        - getGaussianMatrix |       0.00ms        0.00ms
        - getCircleKernel   |       0.00ms        0.00ms

    Start to finish: 0000.00ms                               // (bold)

    */

    capture.release();
    return EXIT_SUCCESS;
}