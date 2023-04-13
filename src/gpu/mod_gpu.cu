#include "mod_GPU.hpp"

float *getGaussianMatrix(size_t ksize, double sigma);
uchar *getCircleKernel(size_t diameter);

cv::Mat detectObjectInFrameGPU(const cv::Mat &background, cv::Mat frame)
{
    const int height = frame.rows;
    const int width = frame.cols;
    const int numPixels = height * width;
    const size_t ksize = 15;
    const float *gaussianKernel = getGaussianMatrix(ksize, 2.0);
    const uchar threshold = 20;
    const uchar maxval_tresh = 255;
    const int morphologicalCircleDiameter = 15;
    const uchar *circleKernel = getCircleKernel(morphologicalCircleDiameter);

    uchar3 *d_background;
    uchar3 *d_frame;

    uchar *d_bgd;
    uchar *d_input;
    uchar *d_tmp;

    float *d_gaussianKernel;
    uchar *d_circleKernel;

    cudaMalloc(&d_background, numPixels * sizeof(uchar3));
    cudaMalloc(&d_frame, numPixels * sizeof(uchar3));
    cudaMalloc(&d_input, numPixels * sizeof(uchar));
    cudaMalloc(&d_bgd, numPixels * sizeof(uchar));
    cudaMalloc(&d_gaussianKernel, ksize * ksize * sizeof(float));
    cudaMalloc(&d_circleKernel,
               morphologicalCircleDiameter * morphologicalCircleDiameter
                   * sizeof(uchar));
    cudaMalloc(&d_tmp, height * width * sizeof(uchar));
    cudaMemset(d_tmp, 0, height * width * sizeof(uchar));

    cudaMemcpy(d_background, background.ptr<uchar3>(0),
               numPixels * sizeof(uchar3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_frame, frame.ptr<uchar3>(0), numPixels * sizeof(uchar3),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_gaussianKernel, gaussianKernel, ksize * ksize * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_circleKernel, circleKernel,
               morphologicalCircleDiameter * morphologicalCircleDiameter
                   * sizeof(uchar),
               cudaMemcpyHostToDevice);

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
    // cudaDeviceSynchronize();
    cudaMemcpy(d_input, d_tmp, height * width * sizeof(uchar),
               cudaMemcpyDeviceToDevice);

    cv::Mat output(cv::Size(width, height), CV_8UC1);
    cudaMemcpy(output.ptr<uchar>(0), d_input, numPixels * sizeof(uchar),
               cudaMemcpyDeviceToHost);

    cudaFree(d_background);
    cudaFree(d_frame);
    cudaFree(d_input);
    cudaFree(d_bgd);
    cudaFree(d_gaussianKernel);
    cudaFree(d_circleKernel);
    cudaFree(d_tmp);

    delete[] gaussianKernel;
    delete[] circleKernel;

    return output;
}