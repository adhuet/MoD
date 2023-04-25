#ifndef __UNIONFIND_HPP__
#define __UNIONFIND_HPP__

#include <opencv2/opencv.hpp>
#include <vector>

// Optimized Block-Based Algorithms to Label Connected Components on GPUs
// Allegretti, Bolelli et al

// FIXME uchar* L limits amount of object to 255, but used this way to save
//  a conversion int <-> uchar

/*
Traverse tree to which a belongs, and return the identifier of the tree i.e. the
root
*/
int find(int *L, int a);

/*
Flatten the union-find tree by linking a directly to the root of its tree.
When every tree in the union-fund array exactly matches a connected component,
the Compress procedure can be performed on every node/pixel to produce the
final output image.
*/
void compress(int *L, int a);

/*
Variation of compress, optimized for data parallel environment
*/
int inlineCompress(int *L, int a);

/*
AKA UnionNaive, first calls find to find root for a and b, then sets the smaller
root as the father of the other one, thus joining the two trees into a single
one.
*/
void mergeNaive(int *L, int a, int b);

/*
Uses CUDA atomics to fix the problem in mergeNaive where it was not taken into
account that two threads could reach and modify the same root starting from two
different input nodes.
*/
void merge(int *L, int a, int b);

// Connected Components Labelling, Jack Lawrence-Jones
// class Node
// {
//     int value;
//     Node *parent;
//     int rank;

//     Node(int val);
//     ~Node();
// };

// class UnionFind
// {
// public:
//     std::vector<Node> nodes;

//     UnionFind();
//     ~UnionFind();

//     Node *makeSet(int val);
//     Node *find(const Node &node);
//     void merge(const Node &a, const Node &b);
//     Node *getNode(int val);
// };

#endif