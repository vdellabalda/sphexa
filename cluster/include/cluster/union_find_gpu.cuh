#pragma once

#include "cstone/cuda/cuda_utils.cuh"

namespace unionfind
{

// Find root with path compression
template<class IdType>
__device__ IdType findRootCompressGPU(IdType* clusterId, IdType node)
{
    IdType v = node;
    IdType u;
    IdType w;

    while (true)
    {   
        u = v;
        v = clusterId[u]; // read parent
        w = clusterId[v]; // read grandparent            
        atomicCAS(clusterId + u, v, w); // path compression
        if (v == w) return v; // root found
    }
}

template<class IdType>
__device__ IdType findRootGPU(IdType* clusterId, IdType node, size_t maxSize)
{
    IdType child = node;
    IdType parent;
    
    while (true) {
        parent = clusterId[child];
        if (parent >= maxSize) {
            printf("Warning: Invalid cluster ID %u (max %lu)\n", parent, maxSize);
            return child; // Return current node if we encounter an invalid ID
        }
        if (parent == child) {
            return child; // Found root
        }
        child = parent;
    }
}

template<class IdType>
__device__ void uniteGPU(IdType* clusterId, IdType x, IdType y, size_t maxSize)
{
    if (x == y) return;
    if (x >= maxSize || y >= maxSize) printf("Warning: Invalid cluster ID u=%u v=%u (max %lu)\n", x, y, maxSize);
       
    while (true)
    {   
        IdType u = findRootGPU(clusterId, x, maxSize);
        IdType v = findRootGPU(clusterId, y, maxSize);

        if (u == v) return; // Already in same set
        if (u >= maxSize || v >= maxSize) printf("Warning: Invalid cluster ID u=%u v=%u (max %lu)\n", u, v, maxSize);

        IdType parent = min(u, v);
        IdType child  = max(u, v);
        
        if (atomicCAS(clusterId+child, child, parent) == child) {
            return; // Successfully united
        }
        // Else, retry with updated roots
    }
}

template<class IdType>
__global__ void updateRootGPU(IdType* clusterId, size_t lastBody)
{
    unsigned gid     = blockDim.x * blockIdx.x + threadIdx.x; 
    IdType node  = gid;
    IdType root;

    if (gid < lastBody)
    {
        root = findRootGPU(clusterId, node, lastBody);
        clusterId[gid] = root;
    }
}

template<class LocalIndex>
__global__ void unionFindGpu(
    LocalIndex* clusterId,
    LocalIndex* edgeSrc,
    LocalIndex* edgeDst,
    size_t numEdges,
    size_t numClusters)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numEdges) return;
    uniteGPU(clusterId, edgeSrc[idx], edgeDst[idx], numClusters);
}
}