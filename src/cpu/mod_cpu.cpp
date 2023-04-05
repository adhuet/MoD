#include "mod.hpp"
#include "utils.hpp"

cv::Mat detectObjectInFrameCPU(const cv::Mat &background, cv::Mat frame)
{
    // FIXME
    // Semantically incorrect, s_image only hold 1 channel
    // bgd and image are not correct representations of the two images
    s_image bgd = toPtr(background);
    s_image image = toPtr(frame);

    // (2) Convert both background and frame to grayscale
    grayscale(background, bgd);
    grayscale(frame, image);

    // (3) Smooth the two images
    blur(bgd, bgd, 15, 0.2);
    blur(image, image, 15, 0.2);

    // (4) Compute the difference
    diff(bgd, image, image);
    cv::Mat output = toMat(image);
    return output;

    // (5) Morphological opening
    morphOpen(image, image, 15);

    // (6) Threshold the image and get connected components bboxes
    treshold(image, image, 20, 255);
    auto bboxes = connectedComponents(image);

    // (7) Draw the bboxes on output
    for (const auto &bbox : bboxes)
        cv::rectangle(frame, bbox, cv::Scalar(0, 0, 255), 2);

    return frame;
}
