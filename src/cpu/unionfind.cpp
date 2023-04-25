#include "unionfind.hpp"

// Optimized Block-Based Algorithms to Label Connected Components on GPUs
// Allegretti, Bolelli et al
int find(int *L, int a)
{
    // while (L[a] != a + 1)
    // a = L[a] - 1;
    while (L[a] != a)
        a = L[a];
    return a;
}

void compress(int *L, int a)
{
    L[a] = find(L, a);
}

int inlineCompress(int *L, int a)
{
    int id = a;
    while (L[a] != a)
    {
        a = L[a];
        L[id] = a;
    }
    return a;
}

void mergeNaive(int *L, int a, int b)
{
    a = find(L, a);
    b = find(L, b);

    if (a < b)
        // L[b] = a + 1;
        L[b] = a;
    else if (b < a)
        // L[a] = b + 1;
        L[a] = b;
}

// void merge(int*L, int a, int b)
// {
//     bool done = false;
//     while (!done)
//     {
//         a = find(L, a);
//         b = find(L, b);
//         if (a < b)
//         {

//         }
//     }
// }

// Connected Components Labelling, Jack Lawrence-Jones
// Node::Node(int val)
//     : value(val)
//     , rank(0)
//     , parent(this)
// {}

// UnionFind::UnionFind()
//     : nodes()
// {}

// Node *UnionFind::makeSet(int val)
// {
//     if (this->getNode(val))
//         return this->getNode(val);

//     Node node(val);
// }

// Node *UnionFind::find(const Node &node)
// {}

// void UnionFind::merge(const Node &a, const Node &b)
// {}

// Node *UnionFind::getNode(int val)
// {}