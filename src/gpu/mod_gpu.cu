#include <iomanip>
#include <sstream>

#include "mod_GPU.hpp"
#include "utils.hpp"

float *getGaussianMatrix(size_t ksize, double sigma);
uchar *getCircleKernel(size_t diameter);

int renderObjectsInCaptureGPU(cv::VideoCapture capture)
{
    cv::Mat frame;
    cv::Mat background;
    capture >> background;
    if (background.empty())
    {
        std::cerr << "First frame of video (background) is empty!" << std::endl;
        return -1;
    }

    // Initialize all important constants
    const int block_width = 32;
    const int height = background.rows;
    const int width = background.cols;
    const int numPixels = height * width;
    const size_t ksize = 15;
    const float sigma = 2.0;
    const float *gaussianKernel = getGaussianMatrix(ksize, sigma);
    const uchar threshold = 20;
    const uchar maxval_tresh = 255;
    const int morphological_circle_diameter = 15;
    const uchar *circleKernel = getCircleKernel(morphological_circle_diameter);
    dim3 blockDim(block_width, block_width);
    dim3 gridDim(int(ceil((float)width / block_width)),
                 int(ceil((float)height / block_width)));
    const size_t blur_tile_width = blockDim.x - ksize + 1;
    dim3 blurGridDim(int(ceil((float)width / blur_tile_width)),
                     int(ceil((float)height / blur_tile_width)));

    // Host buffers used during computation
    int *labels = new int[numPixels]; // Holds the final CCL symbollic image
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
    // Process background
    grayscaleGPU<<<gridDim, blockDim>>>(d_background, d_bgd, height, width);
    blurTiledGPU<<<blurGridDim, blockDim,
                   block_width * block_width * sizeof(uchar)>>>(
        d_bgd, d_bgd, height, width, d_gaussianKernel, ksize);
    // We do not this the original background buffer, so we release memory as
    // soon as possible
    cudaFree(d_background);

    // Rest of the allocations for the frame processing
    cudaMalloc(&d_frame, numPixels * sizeof(uchar3));
    cudaMalloc(&d_input, numPixels * sizeof(uchar));
    cudaMalloc(&d_swap, numPixels * sizeof(uchar));
    cudaMalloc(&d_labels, numPixels * sizeof(int));

    // Some cuda events for simple fps timing
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0.0f;

    // Use the capture and process frame by frame
    cv::namedWindow("GPU", cv::WINDOW_AUTOSIZE);
    for (;;)
    {
        // Get the frame
        capture >> frame;
        // End if we reach end of video
        if (frame.empty())
            break;

        // Simple fps timer
        cudaEventRecord(start, 0);

        // Copy the frame to allocated buffer
        cudaMemcpy(d_frame, frame.ptr<uchar3>(0), numPixels * sizeof(uchar3),
                   cudaMemcpyHostToDevice);

        // Launch the algorithm ***********************************************
        // The follwing are done in place
        grayscaleGPU<<<gridDim, blockDim>>>(d_frame, d_input, height, width);

        blurTiledGPU<<<blurGridDim, blockDim,
                       block_width * block_width * sizeof(uchar)>>>(
            d_input, d_input, height, width, d_gaussianKernel, ksize);
        diffGPU<<<gridDim, blockDim>>>(d_bgd, d_input, d_input, height, width);
        thresholdGPU<<<gridDim, blockDim>>>(d_input, d_input, height, width,
                                            threshold, maxval_tresh);

        // We need the swap array here
        //      Opening: dilate + erode
        dilateGPU<<<gridDim, blockDim>>>(d_input, d_swap, height, width,
                                         d_circleKernel,
                                         morphological_circle_diameter);
        erodeGPU<<<gridDim, blockDim>>>(d_swap, d_input, height, width,
                                        d_circleKernel,
                                        morphological_circle_diameter);
        // cudaMemcpy(d_input, d_swap, numPixels * sizeof(uchar),
        // cudaMemcpyDeviceToDevice);
        //      Closing: erode + dilate
        // erodeGPU<<<gridDim, blockDim>>>(d_input, d_swap, height, width,
        //                                 d_circleKernel,
        //                                 morphological_circle_diameter);
        // dilateGPU<<<gridDim, blockDim>>>(d_swap, d_input, height, width,
        //                                  d_circleKernel,
        //                                  morphological_circle_diameter);

        // CCL
        initCCL<<<gridDim, blockDim>>>(d_input, d_labels, height, width);
        mergeCCL<<<gridDim, blockDim>>>(d_input, d_labels, height, width);
        compressCCL<<<gridDim, blockDim>>>(d_input, d_labels, height, width);

        // Get the labels on the host
        cudaMemcpy(labels, d_labels, numPixels * sizeof(int),
                   cudaMemcpyDeviceToHost);

        // Get the bboxes
        bboxes = getBoundingBoxes(labels, width, height);
        // Algorithm ends here ************************************************
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        float fps = 1.0f / (milliseconds / 1000.0f);
        std::ostringstream ostream;
        ostream << "Fps: " << std::fixed << std::setprecision(2) << fps;
        std::string fps_string = ostream.str();

        // Render the new frame with the bboxes
        cv::Mat output;
        frame.copyTo(output);
        for (const auto &bbox : bboxes)
            cv::rectangle(output, bbox, cv::Scalar(0, 0, 255), 2);
        cv::putText(output, fps_string, cv::Point(10, 10),
                    cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 2);

        cv::Mat concat(output.rows, frame.cols + output.cols, output.type());
        frame.copyTo(concat(cv::Rect(0, 0, frame.cols, frame.rows)));
        output.copyTo(
            concat(cv::Rect(frame.cols, 0, output.cols, output.rows)));

        cv::imshow("GPU", concat);
        if (cv::waitKey(20) >= 0)
            break;
    }

    cudaEventDestroy(stop);
    cudaEventDestroy(start);

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
    return 0;
}

std::vector<cv::Rect> detectObjectInFrameGPU(const cv::Mat &background,
                                             const cv::Mat &frame)
{
    const int height = frame.rows;
    const int width = frame.cols;
    const int numPixels = height * width;
    const size_t ksize = 15;
    const float *gaussianKernel = getGaussianMatrix(ksize, 2.0);
    const uchar threshold = 20;
    const uchar maxval_tresh = 255;
    const int morphological_circle_diameter = 15;
    const uchar *circleKernel = getCircleKernel(morphological_circle_diameter);

    uchar3 *d_background;
    uchar3 *d_frame;

    uchar *d_bgd;
    uchar *d_input;
    uchar *d_tmp;

    float *d_gaussianKernel;
    uchar *d_circleKernel;

    CUDA_WARN(cudaMalloc(&d_background, numPixels * sizeof(uchar3)));
    CUDA_WARN(cudaMalloc(&d_frame, numPixels * sizeof(uchar3)));
    CUDA_WARN(cudaMalloc(&d_input, numPixels * sizeof(uchar)));
    CUDA_WARN(cudaMalloc(&d_bgd, numPixels * sizeof(uchar)));
    CUDA_WARN(cudaMalloc(&d_gaussianKernel, ksize * ksize * sizeof(float)));
    CUDA_WARN(cudaMalloc(&d_circleKernel,
                         morphological_circle_diameter
                             * morphological_circle_diameter * sizeof(uchar)));
    CUDA_WARN(cudaMalloc(&d_tmp, height * width * sizeof(uchar)));
    CUDA_WARN(cudaMemset(d_tmp, 0, height * width * sizeof(uchar)));

    CUDA_WARN(cudaMemcpy(d_background, background.ptr<uchar3>(0),
                         numPixels * sizeof(uchar3), cudaMemcpyHostToDevice));
    CUDA_WARN(cudaMemcpy(d_frame, frame.ptr<uchar3>(0),
                         numPixels * sizeof(uchar3), cudaMemcpyHostToDevice));
    CUDA_WARN(cudaMemcpy(d_gaussianKernel, gaussianKernel,
                         ksize * ksize * sizeof(float),
                         cudaMemcpyHostToDevice));
    CUDA_WARN(cudaMemcpy(d_circleKernel, circleKernel,
                         morphological_circle_diameter
                             * morphological_circle_diameter * sizeof(uchar),
                         cudaMemcpyHostToDevice));

    dim3 blockDim(32, 32);
    dim3 gridDim(int(ceil((float)width / blockDim.x)),
                 int(ceil((float)height / blockDim.y)));

    grayscaleGPU<<<gridDim, blockDim>>>(d_background, d_bgd, height, width);
    grayscaleGPU<<<gridDim, blockDim>>>(d_frame, d_input, height, width);

    blurGPU<<<gridDim, blockDim>>>(d_bgd, d_bgd, height, width,
                                   d_gaussianKernel, ksize);
    blurGPU<<<gridDim, blockDim>>>(d_input, d_input, height, width,
                                   d_gaussianKernel, ksize);

    diffGPU<<<gridDim, blockDim>>>(d_bgd, d_input, d_input, height, width);

    thresholdGPU<<<gridDim, blockDim>>>(d_input, d_input, height, width,
                                        threshold, maxval_tresh);

    dilateGPU<<<gridDim, blockDim>>>(d_input, d_tmp, height, width,
                                     d_circleKernel, ksize);
    erodeGPU<<<gridDim, blockDim>>>(d_tmp, d_input, height, width,
                                    d_circleKernel, ksize);
    CUDA_WARN(cudaMemcpy(d_input, d_tmp, height * width * sizeof(uchar),
                         cudaMemcpyDeviceToDevice));
    // cudaDeviceSynchronize();
    int *d_labelled;
    cudaMalloc(&d_labelled, height * width * sizeof(int));

    initCCL<<<gridDim, blockDim>>>(d_input, d_labelled, height, width);
    mergeCCL<<<gridDim, blockDim>>>(d_input, d_labelled, height, width);
    compressCCL<<<gridDim, blockDim>>>(d_input, d_labelled, height, width);

    CUDA_WARN(cudaDeviceSynchronize());

    std::vector<cv::Rect> bboxes;

    // connectedComponentsGPU(d_input, d_labelled, height, width, gridDim,
    //                        blockDim);
    int *labelled = new int[height * width];
    CUDA_WARN(cudaMemcpy(labelled, d_labelled, height * width * sizeof(int),
                         cudaMemcpyDeviceToHost));
    bboxes = getBoundingBoxes(labelled, width, height);
    cudaFree(d_labelled);
    cudaFree(d_background);
    cudaFree(d_frame);
    cudaFree(d_input);
    cudaFree(d_bgd);
    cudaFree(d_gaussianKernel);
    cudaFree(d_circleKernel);
    cudaFree(d_tmp);

    delete[] gaussianKernel;
    delete[] circleKernel;
    delete[] labelled;

    // return labelled;
    return bboxes;
    // return output.toCVMat();
}