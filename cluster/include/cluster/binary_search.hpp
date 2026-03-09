#pragma once

#include "cstone/cuda/annotation.hpp"
#include "cstone/util/tuple.hpp"


namespace binary_search
{

// GPU binary search
template<class KeyType>
HOST_DEVICE_FUN auto binarySearch(
    KeyType searchKey,
    const KeyType* keys,
    size_t numKeys
)
{
    size_t left = 0, right = numKeys;
    bool found = false;
    size_t searchId = 0;
    
    while (left < right) {
        size_t mid = (left + right) / 2;
        if (keys[mid] == searchKey) {
            searchId = mid;
            found = true;
            break;
        } else if (keys[mid] < searchKey) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    return util::tuple<bool, size_t>{found, searchId};
}


template<class KeyType>
HOST_DEVICE_FUN auto binaryFind(
    KeyType searchKey,
    const KeyType* keys,
    size_t numKeys
)
{
    size_t left = 0, right = numKeys;
    bool found = false;
    
    while (left < right) {
        size_t mid = (left + right) / 2;    
        if (keys[mid] == searchKey) {
            found = true;
            break;
        } else if (keys[mid] < searchKey) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    return found;
}
}