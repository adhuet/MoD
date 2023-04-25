#include "mod.hpp"
#include "utils.hpp"

std::vector<cv::Rect> detectObjectInFrameCPU(const cv::Mat &background,
                                             cv::Mat frame)
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

    int *labelled = (int *)malloc(image.width * image.height * sizeof(int));

    // (6) Threshold the image and get connected components bboxes
    connectedComponents(image, labelled);
    SImage output(image.width, image.height, labelled);
    std::vector<cv::Rect> bboxes =
        getBoundingBoxes(labelled, image.width, image.height);
    free(labelled);

    return bboxes;
}
