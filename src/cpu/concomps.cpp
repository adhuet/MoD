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