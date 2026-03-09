#pragma once

#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/set_operations.h>
#include <thrust/partition.h>
#include "cstone/cuda/cub.hpp"
#include "cstone/cuda/errorcheck.cuh"
#include "cstone/util/array.hpp"
#include "cstone/util/tuple.hpp"
#include "cstone/primitives/primitives_gpu.h"
#include "definitions.h"
#include "binary_search.hpp"

namespace cluster
{
using namespace binary_search;

template<class ValueType>
struct thresholdMaskFunctor
{
    ValueType _threshold;
    thresholdMaskFunctor(ValueType threshold) : _threshold(threshold) { }

    __host__ __device__ bool operator()(const ValueType& x) const
    {
        return x >= _threshold;
    }
};


template<class ValueType, class FlagType>
void thresholdMaskGpu(const ValueType* first, const ValueType* last, FlagType* mask, ValueType threshold)
{
    thrust::transform(thrust::device, first, last, mask, thresholdMaskFunctor(threshold));
}

template void thresholdMaskGpu(const uint32_t*, const uint32_t*, bool*, const uint32_t);


template<class KeyType, class IndexType>
struct key32ToKey64Functor
{
    IndexType rank;

    key32ToKey64Functor(IndexType rank_)
        : rank(rank_)
    {}

    __host__ __device__ KeyType operator()(const IndexType& localId) const
    {
        return ((static_cast<KeyType>(rank) << 32) | static_cast<KeyType>(localId));
    };
};

template<class IndexType, class KeyType>
void transformLocalToGlobalClusterKeys(
    const IndexType* localClusterIds,
    KeyType* globalClusterKeys,
    size_t n,
    int rank
)
{   
    thrust::transform(thrust::device, localClusterIds, localClusterIds+n, globalClusterKeys,
        key32ToKey64Functor<KeyType, IndexType>(rank)); 
}
template void transformLocalToGlobalClusterKeys(const uint32_t*, uint64_t*, size_t, int);


template<class KeyType, class IndexType>
__global__ void clusterKeyUpdate(
    KeyType* globalKeys,
    const KeyType* uniqueKeys,
    const IndexType* clusterParents,
    size_t numKeys,
    size_t numParticles)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numParticles) return;

    bool found;
    IndexType localId;
    util::tie(found, localId) = binarySearch(globalKeys[tid], uniqueKeys, numKeys);
    if (!found) { return;}

    IndexType newClusterId = clusterParents[localId];
    KeyType newKey = uniqueKeys[newClusterId];
    globalKeys[tid] = newKey;
    return;
}


template<class KeyType, class FlagType, class IdType, class EdgeType>
__global__ void constructEdgesGpu(const IdType* localClusterIds, const KeyType* globalClusterKeys, const FlagType* flagged, const IdType* compactIndex, 
                                size_t nHaloStart, EdgeType* edges, size_t numParticlesHalos, size_t numParticles, int rank)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticlesHalos) {return;}

    if (idx >= nHaloStart) { idx += numParticles; }
    if (flagged[idx] == 0) { return; }
    unsigned edgeIdx = compactIndex[idx];
    
    IdType localId = localClusterIds[idx];
    KeyType localKey = ((static_cast<KeyType>(rank) << 32) | static_cast<KeyType>(localId));

    if (localKey < globalClusterKeys[idx])
    {
        edges[edgeIdx][0] = localKey;
        edges[edgeIdx][1] = globalClusterKeys[idx];
    }
    else
    {
        edges[edgeIdx][0] = globalClusterKeys[idx];
        edges[edgeIdx][1] = localKey;
    }
}


template<class EdgeType, class KeyType>
__global__ void flattenEdgesGpu(EdgeType* edges, size_t n, KeyType* flatEdges)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {return;}

    flatEdges[2*idx]   = edges[idx][0];
    flatEdges[2*idx+1] = edges[idx][1];
}

template<class KeyType>
__global__ void splitEdgesGpu(KeyType* flatEdges, size_t n, KeyType* edgeSrc, KeyType* edgeDst)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {return;}

    edgeSrc[idx] = flatEdges[2*idx];
    edgeDst[idx] = flatEdges[2*idx+1];
}


template<class ClusterKeyType, class IdType>
__global__ void computeClusterOwnershipGpu(
    const ClusterKeyType* uniqueKeys,
    IdType* ownership,
    size_t numUniqueKeys
)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numUniqueKeys) {return;}

    ownership[idx] = getRankFromClusterKey(uniqueKeys[idx]);
}


// Cluster remapping with binary search
template<class ClusterKeyType, class ClusterIdType>
__global__ void directClusterRemapping(
    const ClusterKeyType* globalClusterKeys,
    ClusterIdType* finalClusterIds,
    const ClusterIdType* selectFlags,
    const ClusterKeyType* sortedKeys,
    const ClusterIdType* sortedIds,
    size_t numParticles,
    size_t numSortedKeys)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numParticles) return;
    if (selectFlags[tid] == 0) 
        {
            finalClusterIds[tid] = 0;
            return;
        }

    ClusterKeyType myKey = globalClusterKeys[tid];
    size_t left = 0, right = numSortedKeys;
    bool found = false;
    ClusterIdType compactId = 0;
    size_t mid;
    ClusterKeyType midKey;
    
    while (left < right) {
        mid = (left + right) / 2;
        midKey = sortedKeys[mid];
        
        if (midKey == myKey) {
            compactId = sortedIds[mid];
            found = true;
            break;
        } else if (midKey < myKey) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    finalClusterIds[tid] = compactId;
}


template<class KeyType, class ValueType>
std::pair<KeyType*, ValueType*> setIntersectionByKeyGpu(
    KeyType* keysFirst1,
    size_t numKeys1,
    KeyType* keysFirst2,
    size_t numKeys2,
    ValueType* valuesFirst,
    KeyType* keysOut,
    ValueType* valuesOut
)
{
    std::pair<KeyType*, ValueType*> new_end = thrust::set_intersection_by_key(
        thrust::device,
        keysFirst1, keysFirst1 + numKeys1,
        keysFirst2, keysFirst2 + numKeys2,
        valuesFirst,
        keysOut,
        valuesOut);
    return new_end;
}
template std::pair<uint64_t*, uint32_t*> setIntersectionByKeyGpu(
    uint64_t*,
    size_t,
    uint64_t*,
    size_t,
    uint32_t*,
    uint64_t*,
    uint32_t*
);


template<class KeyType, class ValueType>
std::pair<KeyType*, ValueType*> setDifferenceByKeyGpu(
    KeyType* keysFirst1,
    size_t numKeys1,
    KeyType* keysFirst2,
    size_t numKeys2,
    ValueType* valuesFirst1,
    ValueType* valuesFirst2,
    KeyType* keysOut,
    ValueType* valuesOut
)
{
    std::pair<KeyType*, ValueType*> new_end = thrust::set_difference_by_key(
        thrust::device,
        keysFirst1, keysFirst1 + numKeys1,
        keysFirst2, keysFirst2 + numKeys2,
        valuesFirst1,
        valuesFirst2,
        keysOut,
        valuesOut);
    return new_end;
}
template std::pair<uint64_t*, uint32_t*> setDifferenceByKeyGpu(
    uint64_t*,
    size_t,
    uint64_t*,
    size_t,
    uint32_t*,
    uint32_t*,
    uint64_t*,
    uint32_t*
);

template<class KeyType, class ValueType>
struct clusterSendFlag
{
    ValueType threshold;
    KeyType* nonLocalKeys;
    size_t numNonLocalKeys;
    clusterSendFlag(ValueType threshold_, KeyType* nonLocalKeys_, size_t numNonLocalKeys_) 
        : threshold(threshold_), nonLocalKeys(nonLocalKeys_), numNonLocalKeys(numNonLocalKeys_) {}

    __host__ __device__ bool operator()(const thrust::tuple<KeyType, ValueType>& t) const
    {
        KeyType key = thrust::get<0>(t);
        ValueType count = thrust::get<1>(t);
        bool isNonLocal = binaryFind(key, nonLocalKeys, numNonLocalKeys);
        return isNonLocal || (count >= threshold);
    };
};

template<class KeyType, class ValueType>
std::pair<KeyType*, ValueType*> selectSendClustersGpu(KeyType* clusterKeys, ValueType* counts, KeyType* nonLocalKeys, ValueType threshold, size_t numClusters, size_t numNonLocalClusters)
{
    thrust::zip_iterator<thrust::tuple<KeyType*, ValueType*>> first = thrust::make_zip_iterator(thrust::make_tuple(clusterKeys, counts));
    thrust::zip_iterator<thrust::tuple<KeyType*, ValueType*>> last = thrust::make_zip_iterator(thrust::make_tuple(clusterKeys + numClusters, counts + numClusters));
    thrust::zip_iterator<thrust::tuple<KeyType*, ValueType*>> newIt = thrust::partition(thrust::device, first, last, clusterSendFlag<KeyType, ValueType>(threshold, nonLocalKeys, numNonLocalClusters));
    return std::make_pair(thrust::get<0>(newIt.get_iterator_tuple()), thrust::get<1>(newIt.get_iterator_tuple()));
}
template std::pair<uint64_t*, uint32_t*> selectSendClustersGpu(uint64_t*, uint32_t*, uint64_t*, uint32_t, size_t, size_t);

template<class CompareType>
struct GreaterThan
{
    CompareType compare;

    __host__ __device__ __forceinline__
    GreaterThan(CompareType compare) : compare(compare) {}

    __host__ __device__ __forceinline__
    bool operator()(const CompareType &a) const {
        return (a > compare);
    }
};

template<class T, class FlagType, class StorageType>
size_t selectByThresholdGpu(const T* input, const FlagType* flags, size_t numElements, FlagType threshold, T* output, StorageType* d_temp_storage, size_t numElementsStorage)
{
    // Determine temporary device storage requirements
    size_t tempStorageBytes = sizeof(StorageType)*numElementsStorage;
    size_t   temp_storage_bytes = 0;
    size_t*   d_num_selected_out;
    checkGpuErrors(cudaMalloc(&d_num_selected_out, sizeof(size_t)));
    checkGpuErrors(cub::DeviceSelect::FlaggedIf(
        nullptr, temp_storage_bytes,
        input, flags, output, d_num_selected_out, numElements, GreaterThan<T>(threshold)));
    if (tempStorageBytes < temp_storage_bytes) { throw std::runtime_error("temp storage too small\n"); };
    
    checkGpuErrors(cub::DeviceSelect::FlaggedIf(
        d_temp_storage, temp_storage_bytes,
        input, flags, output, d_num_selected_out, numElements, GreaterThan<T>(threshold)));

    size_t numSelected;
    checkGpuErrors(cudaMemcpy(&numSelected, d_num_selected_out, sizeof(size_t), cudaMemcpyDeviceToHost));

    checkGpuErrors(cudaFree(d_num_selected_out));
    return numSelected;
}
template size_t selectByThresholdGpu(const unsigned*, const unsigned*, size_t, unsigned, unsigned*, uint32_t*, size_t);
template size_t selectByThresholdGpu(const long unsigned*, const unsigned*, size_t, unsigned, long unsigned*, uint32_t*, size_t);
template size_t selectByThresholdGpu(const unsigned*, const unsigned*, size_t, unsigned, unsigned*, uint64_t*, size_t);
template size_t selectByThresholdGpu(const long unsigned*, const unsigned*, size_t, unsigned, long unsigned*, uint64_t*, size_t);


template<class IdType>
struct FlagRootFunctor
{
    const IdType* parents;
        FlagRootFunctor(const IdType* parents_) : parents(parents_) {}

    __host__ __device__ bool operator()(IdType idx) const
        { return (idx == parents[idx]) ? 1 : 0; }
};

template<class KeyType, class IdType, class StorageType>
size_t selectRootsGpu(const KeyType* uniqueKeys, const IdType* parents, size_t numKeys, KeyType* roots, StorageType* d_temp_storage, size_t numElementsStorage)
{
    // Determine temporary device storage requirements
    size_t tempStorageBytes = sizeof(StorageType)*numElementsStorage;
    size_t   temp_storage_bytes = 0;
    size_t*   d_num_selected_out;
    checkGpuErrors(cudaMalloc(&d_num_selected_out, sizeof(size_t)));
    checkGpuErrors(cub::DeviceSelect::FlaggedIf(
        nullptr, temp_storage_bytes,
        uniqueKeys, parents, roots, d_num_selected_out, numKeys, FlagRootFunctor<IdType>(parents)));
    if (tempStorageBytes < temp_storage_bytes) { throw std::runtime_error("temp storage too small\n"); };
    
    checkGpuErrors(cub::DeviceSelect::FlaggedIf(
        d_temp_storage, temp_storage_bytes,
        uniqueKeys, parents, roots, d_num_selected_out, numKeys, FlagRootFunctor<IdType>(parents)));

    size_t numSelected;
    checkGpuErrors(cudaMemcpy(&numSelected, d_num_selected_out, sizeof(size_t), cudaMemcpyDeviceToHost));

    checkGpuErrors(cudaFree(d_num_selected_out));
    return numSelected;
}
template size_t selectRootsGpu(const uint64_t*, const unsigned*, size_t, uint64_t*, uint32_t*, size_t);
template size_t selectRootsGpu(const uint64_t*, const unsigned*, size_t, uint64_t*, uint64_t*, size_t);


template<class Tinout, class Flag>
uint64_t flagSelectTempStorage(size_t numElements)
{
    size_t temp_storage_bytes = 0;
    checkGpuErrors(cub::DeviceSelect::Flagged(
        nullptr, temp_storage_bytes,
        (Tinout*)nullptr, (Flag*)nullptr, (Tinout*)nullptr, (size_t*)nullptr, numElements));
    return temp_storage_bytes;
}
template uint64_t flagSelectTempStorage<unsigned, unsigned>(size_t);
template uint64_t flagSelectTempStorage<long unsigned, unsigned>(size_t);

template<class Tinout, class FlagType, class StorageType>
void flagSelectGpu(
    const Tinout*     input,
    const FlagType*    flags,
    Tinout*          output,
    size_t         numItems,
    StorageType*          d_temp_storage,
    size_t         numElementsStorage
)
{
    size_t tempStorageBytes = sizeof(StorageType)*numElementsStorage;
    // Determine temporary device storage requirements
    size_t   temp_storage_bytes = 0;
    size_t*   d_num_selected_out;
    checkGpuErrors(cudaMalloc(&d_num_selected_out, sizeof(size_t)));
    checkGpuErrors(cub::DeviceSelect::Flagged(
      nullptr, temp_storage_bytes,
      input, flags, output, d_num_selected_out, numItems));

    // Allocate temporary storage
    //checkGpuErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    if (tempStorageBytes < temp_storage_bytes) { throw std::runtime_error("temp storage too small\n"); };

    // Run selection
    checkGpuErrors(cub::DeviceSelect::Flagged(
      d_temp_storage, temp_storage_bytes,
      input, flags, output, d_num_selected_out, numItems));

    checkGpuErrors(cudaFree(d_num_selected_out));
}
#define FLAG_SELECT_GPU_DB(Tinout, FlagType, StorageType)                                                                     \
        template void flagSelectGpu(const Tinout*, const FlagType*, Tinout*, size_t, StorageType*, size_t)
FLAG_SELECT_GPU_DB(unsigned, unsigned, uint32_t);
FLAG_SELECT_GPU_DB(long unsigned, unsigned, uint32_t);
FLAG_SELECT_GPU_DB(unsigned, unsigned, uint64_t);
FLAG_SELECT_GPU_DB(long unsigned, unsigned, uint64_t);

template<class IndexType>
struct MultipleSequenceFunctor
{
    const IndexType* offsets;
    size_t numSequences;
    
    MultipleSequenceFunctor(const IndexType* offsets_, size_t numSequences_) 
        : offsets(offsets_), numSequences(numSequences_) {}
    
    __host__ __device__ IndexType operator()(IndexType globalIdx) const 
    {
        // Binary search to find sequence index
        IndexType seqIdx = stl::upper_bound(offsets, offsets + numSequences, globalIdx) - offsets - 1;
        return globalIdx - offsets[seqIdx];
    }
};

template<class IndexType>
void multiSequenceGpu(const IndexType* lengths, 
                                     size_t numSequences,
                                     IndexType* output,
                                     size_t totalElements)
{
    // Create offsets array
    cstone::DeviceVector<IndexType> d_offsets(numSequences + 1);
    cstone::exclusiveScanGpu(lengths, lengths + numSequences, rawPtr(d_offsets), IndexType(0));
    
    // Create counting iterator and transform
    auto counting = thrust::make_counting_iterator<IndexType>(0);
    thrust::transform(thrust::device, 
                      counting, counting + totalElements,
                      output,
                      MultipleSequenceFunctor<IndexType>(rawPtr(d_offsets), numSequences));
}

template<class KeyType, class ValueType, class IndexType>
void segmentedSortGpu(
    KeyType* d_keys_in, ValueType* d_values_in,
    size_t totalElements,
    const IndexType* d_offsets, size_t num_segments,
    KeyType* d_keys_buf, ValueType* d_values_buf
)
{
    // Create a set of DoubleBuffers to wrap pairs of device pointers
    cub::DoubleBuffer<KeyType> d_keys(d_keys_in, d_keys_buf);
    cub::DoubleBuffer<ValueType> d_values(d_values_in, d_values_buf);
    // Determine temporary device storage requirements
    void     *d_temp_storage = nullptr;
    size_t   temp_storage_bytes = 0;
    cub::DeviceSegmentedSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys, d_values,
        totalElements, num_segments, d_offsets, d_offsets + 1);
    
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    // Run sorting operation
    cub::DeviceSegmentedSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys, d_values,
        totalElements, num_segments, d_offsets, d_offsets + 1);

    auto* curKeys = d_keys.Current();
    if (curKeys != d_keys_in)
    {
        checkGpuErrors(cudaMemcpy(d_keys_in, curKeys, totalElements * sizeof(KeyType), cudaMemcpyDeviceToDevice));
    }
    auto* curValues = d_values.Current();
    if (curValues != d_values_in)
    {
        checkGpuErrors(cudaMemcpy(d_values_in, curValues, totalElements * sizeof(ValueType), cudaMemcpyDeviceToDevice));
    }
    checkGpuErrors(cudaFree(d_temp_storage));
}
template void segmentedSortGpu(
    float*, uint32_t*,
    size_t,
    const uint32_t*, size_t,
    float*, uint32_t*
);
template void segmentedSortGpu(
    double*, uint32_t*,
    size_t,
    const uint32_t*, size_t,
    double*, uint32_t*
);


struct ConversionFunctor
{
    __host__ __device__ uint64_t operator()(const uint32_t& x) const
    {
        return static_cast<uint64_t>(x);
    }
};
template<class Tin, class Tout>
void convertUint32ToUint64Gpu(const Tin* input, Tout* output, size_t n)
{
    thrust::transform(thrust::device, input, input + n, output,
        ConversionFunctor());
}
template void convertUint32ToUint64Gpu(const uint32_t*, uint64_t*, size_t);

} // namespace cluster