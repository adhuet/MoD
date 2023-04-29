// #include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>

// #include "mod.hpp"
// #include "mod_GPU.hpp"

#define CPU_VERSION 1.0
#define GPU_VERSION 1.0
#define OCV_VERSION 1.0

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

    float test_duration = 0.0; // 15000.01;
    float test_duration2 = 0.0; // 0.01;
    float test_percent = 0.0;

    std::cout << std::setfill('#') << std::setw(75) << "\n";
    std::cout << "Filename: " << argv[1] << std::endl;
    std::cout << "Dimensions: " << capture.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
              << capture.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout << "Framerate: " << capture.get(cv::CAP_PROP_FPS) << "fps"
              << std::endl;
    std::cout << "Total nb of frames: " << capture.get(cv::CAP_PROP_FRAME_COUNT)
              << std::endl;
    std::cout << "Duration: "
              << 1000 * capture.get(cv::CAP_PROP_FRAME_COUNT)
            / capture.get(cv::CAP_PROP_FPS)
              << "ms" << std::endl;

    std::cout << std::setfill('-') << std::setw(75) << "\n";
    std::cout << "OPENCV Bench (v" << OCV_VERSION << "):" << std::endl;
    std::cout << '\r' << "FPS:"
              << "[current_fps]" << std::flush;
    // Run opencv bench
    std::cout << std::endl;
    std::cout << "STEP" << std::setfill(' ')
              << std::setw(24 - std::string("STEP").size()) << "|"
              << std::setw(15) << "FRAME_AVG" << std::setw(17) << "TOTAL"
              << std::setw(17) << "EXEC_TIME" << std::endl;
    std::cout << std::endl;
    std::cout << "grayscale" << std::setfill(' ')
              << std::setw(24 - std::string("grayscale").size()) << "|"
              << std::setw(13) << std::setprecision(7) << test_duration << "ms"
              << std::setw(15) << test_duration2 << "ms" << std::setw(16)
              << std::setprecision(3) << test_percent << "%" << std::endl;
    std::cout << "blur" << std::setfill(' ')
              << std::setw(24 - std::string("blur").size()) << "|"
              << std::setw(13) << std::setprecision(7) << test_duration << "ms"
              << std::setw(15) << test_duration2 << "ms" << std::setw(16)
              << std::setprecision(3) << test_percent << "%" << std::endl;
    std::cout << "diff" << std::setfill(' ')
              << std::setw(24 - std::string("diff").size()) << "|"
              << std::setw(13) << std::setprecision(7) << test_duration << "ms"
              << std::setw(15) << test_duration2 << "ms" << std::setw(16)
              << std::setprecision(3) << test_percent << "%" << std::endl;
    std::cout << "threshold" << std::setfill(' ')
              << std::setw(24 - std::string("threshold").size()) << "|"
              << std::setw(13) << std::setprecision(7) << test_duration << "ms"
              << std::setw(15) << test_duration2 << "ms" << std::setw(16)
              << std::setprecision(3) << test_percent << "%" << std::endl;
    std::cout << "morph" << std::setfill(' ')
              << std::setw(24 - std::string("morph").size()) << "|"
              << std::setw(13) << std::setprecision(7) << test_duration << "ms"
              << std::setw(15) << test_duration2 << "ms" << std::setw(16)
              << std::setprecision(3) << test_percent << "%" << std::endl;
    std::cout << "connectedComps" << std::setfill(' ')
              << std::setw(24 - std::string("connectedComps").size()) << "|"
              << std::setw(13) << std::setprecision(7) << test_duration << "ms"
              << std::setw(15) << test_duration2 << "ms" << std::setw(16)
              << std::setprecision(3) << test_percent << "%" << std::endl;
    std::cout << "bboxes" << std::setfill(' ')
              << std::setw(24 - std::string("bboxes").size()) << "|"
              << std::setw(13) << std::setprecision(7) << test_duration << "ms"
              << std::setw(15) << test_duration2 << "ms" << std::setw(16)
              << std::setprecision(3) << test_percent << "%" << std::endl;

    std::cout << std::endl
              << "Start to finish: \033[1m" << test_duration << "ms\033[0m"
              << std::endl;

    std::cout << std::setfill('-') << std::setw(75) << "\n";
    std::cout << "CPU Bench (v" << CPU_VERSION << "):" << std::endl;
    std::cout << '\r' << "FPS:"
              << "[current_fps]" << std::flush;
    // Run cpu bench
    std::cout << std::endl;
    std::cout << "STEP" << std::setfill(' ')
              << std::setw(24 - std::string("STEP").size()) << "|"
              << std::setw(15) << "FRAME_AVG" << std::setw(17) << "TOTAL"
              << std::setw(17) << "EXEC_TIME" << std::endl;
    std::cout << std::endl;
    std::cout << "grayscale" << std::setfill(' ')
              << std::setw(24 - std::string("grayscale").size()) << "|"
              << std::setw(13) << std::setprecision(7) << test_duration << "ms"
              << std::setw(15) << test_duration2 << "ms" << std::setw(16)
              << std::setprecision(3) << test_percent << "%" << std::endl;
    std::cout << "blur" << std::setfill(' ')
              << std::setw(24 - std::string("blur").size()) << "|"
              << std::setw(13) << std::setprecision(7) << test_duration << "ms"
              << std::setw(15) << test_duration2 << "ms" << std::setw(16)
              << std::setprecision(3) << test_percent << "%" << std::endl;
    std::cout << "diff" << std::setfill(' ')
              << std::setw(24 - std::string("diff").size()) << "|"
              << std::setw(13) << std::setprecision(7) << test_duration << "ms"
              << std::setw(15) << test_duration2 << "ms" << std::setw(16)
              << std::setprecision(3) << test_percent << "%" << std::endl;
    std::cout << "threshold" << std::setfill(' ')
              << std::setw(24 - std::string("threshold").size()) << "|"
              << std::setw(13) << std::setprecision(7) << test_duration << "ms"
              << std::setw(15) << test_duration2 << "ms" << std::setw(16)
              << std::setprecision(3) << test_percent << "%" << std::endl;
    std::cout << "morph" << std::setfill(' ')
              << std::setw(24 - std::string("morph").size()) << "|"
              << std::setw(13) << std::setprecision(7) << test_duration << "ms"
              << std::setw(15) << test_duration2 << "ms" << std::setw(16)
              << std::setprecision(3) << test_percent << "%" << std::endl;
    std::cout << "connectedComps" << std::setfill(' ')
              << std::setw(24 - std::string("connectedComps").size()) << "|"
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

    std::cout << std::endl
              << "Start to finish: \033[1m" << test_duration << "ms\033[0m"
              << std::endl;

    std::cout << std::setfill('-') << std::setw(75) << "\n";
    std::cout << "GPU Bench (v" << GPU_VERSION << "):" << std::endl;
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

    std::cout << std::endl
              << "Start to finish: \033[1m" << test_duration << "ms\033[0m"
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