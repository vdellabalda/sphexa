#pragma once

#include "cstone/cuda/annotation.hpp"
#include "cstone/util/tuple.hpp"


namespace binary_search
{

// GPU binary search
template<class KeyType, class IdType>
HOST_DEVICE_FUN auto binarySearch(
    KeyType searchKey,
    const KeyType* keys,
    const IdType* ids,
    size_t numKeys
)
{
    size_t left = 0, right = numKeys;
    bool found = false;
    IdType searchId = 0;
    
    while (left < right) {
        size_t mid = (left + right) / 2;
        if (keys[mid] == searchKey) {
            searchId = ids[mid];
            found = true;
            break;
        } else if (keys[mid] < searchKey) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    return util::tuple<bool, IdType>{found, searchId};
}
}