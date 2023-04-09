#include "mod.hpp"
#include "utils.hpp"

cv::Mat detectObjectInFrameCPU(const cv::Mat &background, cv::Mat frame)
{
    // FIXME
    // Semantically incorrect, SImage only hold 1 channel
    // bgd and image are not correct representations of the two images
    SImage bgd(background);
    SImage image(frame);

    // (2) Convert both background and frame to grayscale
    grayscale(background, bgd);
    grayscale(frame, image);

    // (3) Smooth the two images
    blur(bgd, bgd, 15, 0.2);
    blur(image, image, 15, 0.2);

    // (4) Compute the difference
    diff(bgd, image, image);

    // Compute the threshold before to ease our life with thresholding
    treshold(image, image, 20, 255);

    // (5) Morphological opening
    morphOpen(image, image, 15);
    return image.toCVMat();

    // (6) Threshold the image and get connected components bboxes
    auto bboxes = connectedComponents(image);

    // (7) Draw the bboxes on output
    for (const auto &bbox : bboxes)
        cv::rectangle(frame, bbox, cv::Scalar(0, 0, 255), 2);

    return frame;
}
