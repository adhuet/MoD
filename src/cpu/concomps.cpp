#include "mod.hpp"
#include "unionfind.hpp"
#include "utils.hpp"

// Get the neighbours of the (x, y) pixel with respect to the Rosenfeld mask:
// p q r
// s x
std::vector<int> rosenfeldNeighbours(int *L, int width, int x, int y)
{
    std::vector<int> neighbours;

    // Not on top edge
    if (y > 0)
    {
        if (x > 0 && L[(y - 1) * width + (x - 1)] != 0)
            neighbours.push_back(L[(y - 1) * width + (x - 1)]); // p

        if (L[(y - 1) * width + x] != 0)
            neighbours.push_back(L[(y - 1) * width + x]); // q

        if (x < width - 1 && L[(y - 1) * width + (x + 1)] != 0)
            neighbours.push_back(L[(y - 1) * width + (x + 1)]);
    }

    // Not on left edge
    if (x > 0 && L[y * width + (x - 1)] != 0)
        neighbours.push_back(L[y * width + (x - 1)]);

    return neighbours;
}

// Using union-find
// Following the three kernels implementation of Oliveira et al.
void connectedComponents(const SImage &src, int *dst)
{
    int height = src.height;
    int width = src.width;

    // Init
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            if (src.data[y * width + x] != 0)
                dst[y * width + x] = y * width + x;
            else
                dst[y * width + x] = 0;
        }
    }

    // Merge
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int index = y * width + x;
            // Work only on foreground pixels
            if (dst[index] == 0)
                continue;

            std::vector<int> neighbours = rosenfeldNeighbours(dst, width, x, y);
            for (int label : neighbours)
                mergeNaive(dst, index, label);
        }
    }

    // Compress
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
            compress(dst, y * width + x);
    }
}

std::vector<cv::Rect> getBoundingBoxes(int *L, int width, int height)
{
    std::unordered_map<int, struct SBox> boxes;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            // We are on a label
            if (L[y * width + x] != 0)
            {
                int id = L[y * width + x];

                // The box with this label already exists
                if (boxes.count(id) != 0)
                {
                    // Update the SBox attributes
                    boxes[id].max_x = std::max(boxes[id].max_x, x);
                    boxes[id].min_x = std::min(boxes[id].min_x, x);
                    boxes[id].max_y = std::max(boxes[id].max_y, y);
                    boxes[id].min_y = std::min(boxes[id].min_y, y);
                }
                else // Otherwise create it
                    boxes[id] = { x, y, x, y };
            }
        }
    }

    std::vector<cv::Rect> vec;
    for (auto it = boxes.begin(); it != boxes.end(); it++)
        vec.push_back(it->second.toCVRect());
    return vec;
}