// #include <cuda_runtime.h>
#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "mod.hpp"
#include "utils.hpp"

// #include "mod_GPU.hpp"

#define CPU_VERSION 1.0
#define GPU_VERSION 1.0
#define OCV_VERSION 1.0

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
    double get_gaussian_circle;
    double gpu_mem_management;
};

typedef struct BM_times BM_times;

void erodeBinary255(const SImage &src, SImage &dst, uchar *kernel,
                    size_t ksize);
void dilateBinary255(const SImage &src, SImage &dst, uchar *kernel,
                     size_t ksize);

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
        timers.get_gaussian_circle += static_cast<double>(
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
        std::cout << '\r' << "FPS: " << std::setprecision(4)
                  << 1
                / (static_cast<double>(
                       std::chrono::duration_cast<std::chrono::milliseconds>(
                           end - start)
                           .count())
                   / 1000.0f)
                  << std::flush;
    }
    std::cout << std::endl;
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
        std::cout << '\r' << "FPS: " << std::setprecision(4)
                  << 1
                / (static_cast<double>(
                       std::chrono::duration_cast<std::chrono::microseconds>(
                           end - start)
                           .count())
                   / 1000000.0f)
                  << std::flush;
    }
    std::cout << std::endl;
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

    float test_duration = 0.0; // 15000.01;
    float test_duration2 = 0.0; // 0.01;
    float test_percent = 0.0;

    std::cout << std::setfill('#') << std::setw(75) << "\n";
    std::cout << "Filename: " << argv[1] << std::endl;
    std::cout << "Dimensions: " << capture.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
              << capture.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout << "Framerate: " << capture.get(cv::CAP_PROP_FPS) << "fps"
              << std::endl;
    std::cout << "Total nb of frames: " << nb_frames << std::endl;
    std::cout << "Duration: "
              << 1000 * capture.get(cv::CAP_PROP_FRAME_COUNT)
            / capture.get(cv::CAP_PROP_FPS)
              << "ms" << std::endl;

    std::cout << std::setfill('-') << std::setw(75) << "\n";
    std::cout << "OPENCV Bench (v" << OCV_VERSION << "):" << std::endl;

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
              << "Start to finish: \033[1m" << duration / 1000.0f << "ms\033[0m"
              << std::endl;

    capture.set(cv::CAP_PROP_POS_FRAMES, 0);
    std::cout << std::setfill('-') << std::setw(75) << "\n";
    std::cout << "CPU Bench (v" << CPU_VERSION << "):" << std::endl;

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
              << timers.get_gaussian_circle / nb_frames << "μs" << std::setw(15)
              << timers.get_gaussian_circle << "μs" << std::setw(16)
              << std::setprecision(3)
              << timers.get_gaussian_circle / (duration * 1000.0f) * 100 << "%"
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
    std::cout << "Other" << std::setfill(' ')
              << std::setw(24 - std::string("Other").size()) << "|"
              << std::setw(13) << std::setprecision(7) << test_duration << "ms"
              << std::setw(15) << test_duration2 << "ms" << std::setw(16)
              << std::setprecision(3) << test_percent << "%" << std::endl;

    std::cout << std::endl
              << "Start to finish: \033[1m" << duration / 1000.0f << "s\033[0m"
              << std::endl;

    std::cout << std::setfill('-') << std::setw(75) << "\n";
    std::cout << "GPU Bench (v" << GPU_VERSION << "):" << std::endl;
    /*
    std::cout << "blockDim: "
              << "[block_width]"
              << "x"
              << "[block_height]" << std::endl;
    std::cout << "gridDim: "
              << "[grid_width]"
              << "x"
              << "[grid_height]" << std::endl;
    std::cout << '\r' << "FPS:"
              << "[current_fps]" << std::flush;
    // Run gpu bench
    std::cout << std::endl;
    std::cout << "STEP" << std::setfill(' ')
              << std::setw(24 - std::string("STEP").size()) << "|"
              << std::setw(15) << "FRAME_AVG" << std::setw(17) << "TOTAL"
              << std::setw(17) << "EXEC_TIME" << std::endl;
    std::cout << std::endl;
    std::cout << "grayscaleGPU" << std::setfill(' ')
              << std::setw(24 - std::string("grayscaleGPU").size()) << "|"
              << std::setw(13) << std::setprecision(7) << test_duration << "ms"
              << std::setw(15) << test_duration2 << "ms" << std::setw(16)
              << std::setprecision(3) << test_percent << "%" << std::endl;
    std::cout << "blurGPU" << std::setfill(' ')
              << std::setw(24 - std::string("blurGPU").size()) << "|"
              << std::setw(13) << std::setprecision(7) << test_duration << "ms"
              << std::setw(15) << test_duration2 << "ms" << std::setw(16)
              << std::setprecision(3) << test_percent << "%" << std::endl;
    std::cout << "diffGPU" << std::setfill(' ')
              << std::setw(24 - std::string("diffGPU").size()) << "|"
              << std::setw(13) << std::setprecision(7) << test_duration << "ms"
              << std::setw(15) << test_duration2 << "ms" << std::setw(16)
              << std::setprecision(3) << test_percent << "%" << std::endl;
    std::cout << "thresholdGPU" << std::setfill(' ')
              << std::setw(24 - std::string("thresholdGPU").size()) << "|"
              << std::setw(13) << std::setprecision(7) << test_duration << "ms"
              << std::setw(15) << test_duration2 << "ms" << std::setw(16)
              << std::setprecision(3) << test_percent << "%" << std::endl;
    std::cout << "dilateGPU" << std::setfill(' ')
              << std::setw(24 - std::string("dilateGPU").size()) << "|"
              << std::setw(13) << std::setprecision(7) << test_duration << "ms"
              << std::setw(15) << test_duration2 << "ms" << std::setw(16)
              << std::setprecision(3) << test_percent << "%" << std::endl;
    std::cout << "erodeGPU" << std::setfill(' ')
              << std::setw(24 - std::string("erodeGPU").size()) << "|"
              << std::setw(13) << std::setprecision(7) << test_duration << "ms"
              << std::setw(15) << test_duration2 << "ms" << std::setw(16)
              << std::setprecision(3) << test_percent << "%" << std::endl;
    std::cout << "connectedComps" << std::setfill(' ')
              << std::setw(24 - std::string("connectedComps").size()) << "|"
              << std::setw(13) << std::setprecision(7) << test_duration << "ms"
              << std::setw(15) << test_duration2 << "ms" << std::setw(16)
              << std::setprecision(3) << test_percent << "%" << std::endl;
    std::cout << "  - initCCL" << std::setfill(' ')
              << std::setw(24 - std::string("  - initCCL").size()) << "|"
              << std::setw(13) << std::setprecision(7) << test_duration << "ms"
              << std::setw(15) << test_duration2 << "ms" << std::setw(16)
              << std::setprecision(3) << test_percent << "%" << std::endl;
    std::cout << "  - mergeCCL" << std::setfill(' ')
              << std::setw(24 - std::string("  - mergeCCL").size()) << "|"
              << std::setw(13) << std::setprecision(7) << test_duration << "ms"
              << std::setw(15) << test_duration2 << "ms" << std::setw(16)
              << std::setprecision(3) << test_percent << "%" << std::endl;
    std::cout << "  - compressCCL" << std::setfill(' ')
              << std::setw(24 - std::string("  - compressCCL").size()) << "|"
              << std::setw(13) << std::setprecision(7) << test_duration << "ms"
              << std::setw(15) << test_duration2 << "ms" << std::setw(16)
              << std::setprecision(3) << test_percent << "%" << std::endl;
    std::cout << "bboxes" << std::setfill(' ')
              << std::setw(24 - std::string("bboxes").size()) << "|"
              << std::setw(13) << std::setprecision(7) << test_duration << "ms"
              << std::setw(15) << test_duration2 << "ms" << std::setw(16)
              << std::setprecision(3) << test_percent << "%" << std::endl;
    std::cout << "Other" << std::setfill(' ')
              << std::setw(24 - std::string("Other").size()) << "|"
              << std::setw(13) << std::setprecision(7) << test_duration << "ms"
              << std::setw(15) << test_duration2 << "ms" << std::setw(16)
              << std::setprecision(3) << test_percent << "%" << std::endl;
    std::cout << "  - getGaussianMatrix" << std::setfill(' ')
              << std::setw(24 - std::string("  - getGaussianMatrix").size())
              << "|" << std::setw(13) << std::setprecision(7) << test_duration
              << "ms" << std::setw(15) << test_duration2 << "ms"
              << std::setw(16) << std::setprecision(3) << test_percent << "%"
              << std::endl;
    std::cout << "  - getCircleKernel" << std::setfill(' ')
              << std::setw(24 - std::string("  - getCircleKernel").size())
              << "|" << std::setw(13) << std::setprecision(7) << test_duration
              << "ms" << std::setw(15) << test_duration2 << "ms"
              << std::setw(16) << std::setprecision(3) << test_percent << "%"
              << std::endl;
    std::cout << "Mem. Management" << std::setfill(' ')
              << std::setw(24 - std::string("Mem. Management").size()) << "|"
              << std::setw(13) << std::setprecision(7) << test_duration << "ms"
              << std::setw(15) << test_duration2 << "ms" << std::setw(16)
              << std::setprecision(3) << test_percent << "%" << std::endl;

    std::cout << std::endl
              << "Start to finish: \033[1m" << test_duration / 1000.0f
              << "s\033[0m" << std::endl;
    */
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