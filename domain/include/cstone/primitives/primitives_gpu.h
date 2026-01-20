/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Basic algorithms on the GPU
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <span>
#include <tuple>
#include <cstone/tree/definitions.h>

namespace cstone
{

template<class T>
extern void fillGpu(T* first, T* last, T value);

template<class T>
extern void scaleGpu(T* first, T* last, T value);

template<class T, class Tf>
extern void multiplyElementWiseGpu(T* first, T* last, const Tf* factors);

template<class TS, class TD, class IndexType>
extern void gatherGpu(const IndexType* ordering, size_t numElements, const TS* src, TD* buffer);

//! @brief Lambda to avoid templated functors that would become template-template parameters when passed to functions.
inline auto gatherGpuL = [](std::span<const LocalIndex> ordering, const auto* src, auto* dest)
{ gatherGpu(ordering.data(), ordering.size(), src, dest); };

template<class T, class IndexType>
extern void scatterGpu(const IndexType* ordering, size_t numElements, const T* src, T* buffer);

template<class T, class IndexType>
extern void gatherScatterGpu(const IndexType* gmap, const IndexType* smap, size_t numElements, const T* src, T* buffer);

template<class T>
struct MinMaxGpu
{
    std::tuple<T, T> operator()(const T* first, const T* last);
};

template<class T>
T maxNormSquareGpu(const T* x, const T* y, const T* z, size_t numElements);

template<class T>
extern size_t lowerBoundGpu(const T* first, const T* last, T value);

template<class T, class IndexType>
extern void lowerBoundGpu(const T* first, const T* last, const T* valueFirst, const T* valueLast, IndexType* result);

template<class T1, class T2, class Tout>
extern void sequenceMax(const T1* i1_begin, const T1* i1_end, const T2* i2, Tout* output);

template<class Tin, class Tout>
extern Tout reduceGpu(const Tin* input, size_t numElements, Tout init);

template<class KeyType, class Tin, class Tout>
extern std::pair<KeyType*, Tout*> reduceByKeyGpu(
    const KeyType* keysFirst,
    const KeyType* keysLast,
    const Tin* valuesFirst,
    KeyType* keysOut,
    Tout* valuesOut
);

template<class IndexType>
extern void sequenceGpu(IndexType* input, size_t numElements, IndexType init);

/*! @brief sort range [first:last], using @p keyBuf as temporary storage
 *
 * @param[inout] first   pointer to first element in range
 * @param[inout] last    pointer to last element in range
 * @param[-]     keyBuf  buffer of length last-first for temporary usage
 */
template<class KeyType>
extern void sortGpu(KeyType* first, KeyType* last, KeyType* keyBuf);

template<class EdgeType>
extern void sortGpu(EdgeType* first, EdgeType* last);

//! @brief Determine temporary device storage requirements for sortByKeyGpu
template<class KeyType, class ValueType>
extern uint64_t sortByKeyTempStorage(uint64_t numElements);

template<class KeyType, class ValueType>
extern void
sortByKeyGpu(KeyType* first, KeyType* last, ValueType* values, KeyType* keyBuf, ValueType* valueBuf, void*, uint64_t);

template<class KeyType, class ValueType>
extern void sortByKeyGpu(KeyType* first, KeyType* last, ValueType* values);

template<class KeyType, class ValueType>
extern void sortByKeyDescendGpu(KeyType* first, KeyType* last, ValueType* values);

template<class IndexType, class SumType>
extern void exclusiveScanGpu(const IndexType* first, const IndexType* last, SumType* output, SumType init);

template<class IndexType, class SumType>
extern void inclusiveScanGpu(const IndexType* first, const IndexType* last, SumType* output);

template<class IndexType, class SumType>
void exclusiveScanGpu(const IndexType* first, const IndexType* last, SumType* output)
{
    exclusiveScanGpu(first, last, output, SumType(0));
}

template<class ValueType>
extern size_t countGpu(const ValueType* first, const ValueType* last, ValueType v);

template<class TS, class TD, class S>
extern void selectCopyGpu(const TS* src, LocalIndex n, const S* selectFlags, TD* dest);

template<class TS, class TD, class S>
extern void selectCompactCopyGpu(const TS* src, LocalIndex n, const S* selectFlags, const LocalIndex* index, TD* dest);

template<class EdgeType>
extern EdgeType* uniqueGpu(EdgeType* first, EdgeType* last);

template<class IndexType>
extern size_t uniqueCountGpu(const IndexType* first, const IndexType* last);

template<class KeyType, class IndexType>
extern void runLengthEncodeGpu(const size_t num_items, const KeyType* d_in, KeyType* d_unique_out, IndexType* d_counts_out,
    IndexType* d_num_runs_out);

} // namespace cstone
