#include "mod_GPU.hpp"

cv::Mat detectObjectInFrameGPU(const cv::Mat &background, cv::Mat frame)
{
    const int height = frame.rows;
    const int width = frame.cols;
    const int numPixels = height * width;

    uchar3 *d_background;
    uchar3 *d_frame;

    unsigned char *d_bgd;
    unsigned char *d_input;

    cudaMalloc(&d_background, numPixels * sizeof(uchar3));
    cudaMalloc(&d_frame, numPixels * sizeof(uchar3));
    cudaMalloc(&d_input, numPixels * sizeof(unsigned char));
    cudaMalloc(&d_bgd, numPixels * sizeof(unsigned char));

    cudaMemcpy(d_background, background.ptr<uchar3>(0),
               numPixels * sizeof(uchar3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_frame, frame.ptr<uchar3>(0), numPixels * sizeof(uchar3),
               cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32);
    dim3 gridDim(int(ceil((float)width / blockDim.x)),
                 int(ceil((float)height / blockDim.y)));

    grayscaleGPU<<<gridDim, blockDim>>>(d_background, d_bgd, height, width);
    grayscaleGPU<<<gridDim, blockDim>>>(d_frame, d_input, height, width);

    cv::Mat output(cv::Size(width, height), CV_8UC1);
    cudaMemcpy(output.ptr<unsigned char>(0), d_input,
               numPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_background);
    cudaFree(d_frame);
    cudaFree(d_input);
    cudaFree(d_bgd);

    return output;
}