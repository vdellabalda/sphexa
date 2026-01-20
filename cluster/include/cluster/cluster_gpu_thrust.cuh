#pragma once

#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/set_operations.h>
#include "cstone/cuda/cub.hpp"
#include "cstone/cuda/errorcheck.cuh"
#include "cstone/util/array.hpp"

namespace cluster
{

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

template void thresholdMaskGpu(const unsigned*, const unsigned*, bool*, const unsigned);


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
    const IndexType* localClusterKeys,
    KeyType* globalClusterKeys,
    const size_t n,
    const int rank
)
{   
    thrust::transform(thrust::device, localClusterKeys, localClusterKeys+n, globalClusterKeys,
        key32ToKey64Functor<KeyType, IndexType>(rank)); 
}

template void transformLocalToGlobalClusterKeys(
    const unsigned*, uint64_t*, const unsigned long, const int);

//struct maxCountOwnerFunctor
//{
//    __host__ __device__ unsigned operator()(const util::array<unsigned, 2>& a,
//                                           const util::array<unsigned, 2>& b) const
//    {
//        if (a[0] > b[0])
//            return a[1];
//        return b[1];
//    }
//};
//
//template<class KeyType, class Tin, class Tout>
//std::pair<KeyType*, Tout*> findHaloOwnershipGpu(
//    const KeyType* keysFirst, const KeyType* keysLast,
//    const Tin*     valuesFirst,
//    KeyType*       keysOut,
//    Tout*          valuesOut
//)
//{
//    ::cuda::std::equal_to<int> binary_pred;
//
//    thrust::pair<KeyType*, Tout*> new_end = thrust::reduce_by_key(
//        thrust::device,
//        keysFirst, keysLast,
//        valuesFirst,
//        keysOut,
//        valuesOut,
//        binary_pred,
//        maxCountOwnerFunctor{});
//    return std::pair(new_end.first, new_end.second);
//}
//
//template std::pair<long unsigned*, unsigned*> findHaloOwnershipGpu(
//    const long unsigned*, const long unsigned*,
//    const util::array<unsigned, 2>*,
//    long unsigned*,
//    unsigned*
//);

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

template std::pair<uint64_t*, unsigned*> setIntersectionByKeyGpu(
    uint64_t*,
    size_t,
    uint64_t*,
    size_t,
    unsigned*,
    uint64_t*,
    unsigned*
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

template std::pair<uint64_t*, unsigned*> setDifferenceByKeyGpu(
    uint64_t*,
    size_t,
    uint64_t*,
    size_t,
    unsigned*,
    unsigned*,
    uint64_t*,
    unsigned*
);

template<class Tin, class Flag, class Tout, class IdType>
void flagSelectGpu(
    const Tin*     input,
    const Flag*    flags,
    Tout*          output,
    IdType*        numSelectedOut,
    size_t         numItems
)
{
    // Determine temporary device storage requirements
    void     *d_temp_storage = nullptr;
    size_t   temp_storage_bytes = 0;
    checkGpuErrors(cub::DeviceSelect::Flagged(
      d_temp_storage, temp_storage_bytes,
      input, flags, output, numSelectedOut, numItems));

    // Allocate temporary storage
    checkGpuErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // Run selection
    checkGpuErrors(cub::DeviceSelect::Flagged(
      d_temp_storage, temp_storage_bytes,
      input, flags, output, numSelectedOut, numItems));

    checkGpuErrors(cudaFree(d_temp_storage));
}

template void flagSelectGpu(const unsigned*, const unsigned*, unsigned*, unsigned*, size_t);
template void flagSelectGpu(const long unsigned*, const unsigned*, long unsigned*, unsigned*, size_t);


// Define predicate: select if flag is bigger than or equal to 
template<class Flag>
struct isLargerThanThreshold
    {
        size_t threshold;
        isLargerThanThreshold(size_t threshold_) : threshold(threshold_) {}
        __host__ __device__ bool operator()(const Flag& x) const
        {
            return x >= threshold;
        }
    };

template<class Tin, class Flag, class Tout>
void flagIfSelectGpu(
    const Tin*     input,
    const Flag*    flags,
    Tout*          output,
    int*           numSelectedOut,
    size_t         numItems,
    size_t         clusterThreshold
)
{
    // Determine temporary device storage requirements
    void     *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceSelect::FlaggedIf(
        nullptr,
        temp_storage_bytes,
        input,
        flags,
        output,
        numSelectedOut,
        numItems,
        isLargerThanThreshold<Flag>(clusterThreshold)
    );

    // Allocate temporary storage
    checkGpuErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // Run selection
    cub::DeviceSelect::FlaggedIf(
        nullptr,
        temp_storage_bytes,
        input,
        flags,
        output,
        numSelectedOut,
        numItems,
        isLargerThanThreshold<Flag>(clusterThreshold)
    );
    checkGpuErrors(cudaFree(d_temp_storage));
}
}