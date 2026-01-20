namespace cluster
{
/*! @brief GPU Hash Map for efficient global cluster key to local halo ID mapping
 *  @author Vincente Della Balda <vinc.dellabalda@gmail.com>
 *
 * This hash map uses open addressing with linear probing for collision resolution.
 * It's designed for concurrent read access from GPU threads with atomic operations
 * for safe insertion during parallel construction.
 */
template<class KeyType, class ValueType>
struct GpuHashMap {
    static constexpr KeyType EMPTY_KEY = static_cast<KeyType>(-1);
    static constexpr ValueType EMPTY_VALUE = static_cast<ValueType>(-1);
    
    KeyType* keys;          //!< Array of keys
    ValueType* values;      //!< Array of values
    unsigned capacity;      //!< Hash table capacity (power of 2)
    unsigned mask;          //!< Bitmask for fast modulo (capacity - 1)
    unsigned size;          //!< Current number of elements
    
    __host__ __device__ GpuHashMap() : keys(nullptr), values(nullptr), capacity(0), mask(0), size(0) {}
    
    __host__ __device__ GpuHashMap(KeyType* k, ValueType* v, size_t cap) 
        : keys(k), values(v), capacity(cap), mask(cap - 1), size(0) {}
};

/*! @brief Hash function for cluster keys */
template<class KeyType>
__host__ __device__ size_t hashFunction(KeyType key, size_t mask) {
    // 64-bit golden ratio approximation: 2^64 / φ
    return (key * 0x9e3779b97f4a7c15ull) & mask;
}

/*! @brief Insert key-value pair into GPU hash map with atomic operations
 *
 * @param[inout] hashMap    Hash map to insert into
 * @param[in]    key        Key to insert
 * @param[in]    value      Value to associate with key
 * @return                  true if inserted successfully, false if table is full or key exists
 */
template<class KeyType, class ValueType>
__device__ bool gpuHashMapInsert(GpuHashMap<KeyType, ValueType>& hashMap, KeyType key, ValueType value) {
    if (key == GpuHashMap<KeyType, ValueType>::EMPTY_KEY) return false;
    
    size_t index = hashFunction(key, hashMap.mask);
    size_t original_index = index;
    
    do {
        KeyType existing_key = atomicCAS((unsigned long long*)&hashMap.keys[index], 
                                        (unsigned long long)GpuHashMap<KeyType, ValueType>::EMPTY_KEY, (unsigned long long)key);
        
        if (existing_key == GpuHashMap<KeyType, ValueType>::EMPTY_KEY) {
            // Successfully claimed this slot
            hashMap.values[index] = value;
            return true;
        } else if (existing_key == key) {
            // Key already exists, update value
            hashMap.values[index] = value;
            return true;
        }
        
        index = (index + 1) & hashMap.mask;
    } while (index != original_index);
    
    return false;  // Hash map is full
}

/*! @brief Lookup value by key in GPU hash map
 *
 * @param[in]    hashMap    Hash map to search
 * @param[in]    key        Key to look up
 * @return                  Associated value or EMPTY_VALUE if not found
 */
template<class KeyType, class ValueType>
__device__ ValueType gpuHashMapLookup(const GpuHashMap<KeyType, ValueType>& hashMap, KeyType key) {
    if (key == GpuHashMap<KeyType, ValueType>::EMPTY_KEY) {
        return GpuHashMap<KeyType, ValueType>::EMPTY_VALUE;
    }
    
    size_t index = hashFunction(key, hashMap.mask);
    size_t original_index = index;
    
    do {
        KeyType existing_key = hashMap.keys[index];
        
        if (existing_key == key) {
            return hashMap.values[index];
        } else if (existing_key == GpuHashMap<KeyType, ValueType>::EMPTY_KEY) {
            // Empty slot found, key doesn't exist
            return GpuHashMap<KeyType, ValueType>::EMPTY_VALUE;
        }
        
        // Move to next slot
        index = (index + 1) & hashMap.mask;
    } while (index != original_index);
    
    return GpuHashMap<KeyType, ValueType>::EMPTY_VALUE;  // Not found after full traversal
}

/*! @brief Initialize GPU hash map with empty values
 *
 * @param[inout] hashMap    Hash map to initialize
 */
template<class KeyType, class ValueType>
__global__ void initializeGpuHashMap(GpuHashMap<KeyType, ValueType> hashMap) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= hashMap.capacity) return;

    hashMap.keys[tid] = GpuHashMap<KeyType, ValueType>::EMPTY_KEY;
    hashMap.values[tid] = GpuHashMap<KeyType, ValueType>::EMPTY_VALUE;
    
}

/*! @brief Allocate GPU hash map with specified capacity
 *
 * @param[out]   hashMap    Hash map structure to initialize
 * @param[in]    capacity   Desired capacity (will be rounded up to next power of 2)
 * @return                  cudaError_t indicating success or failure
 */
template<class KeyType, class ValueType>
cudaError_t allocateGpuHashMap(GpuHashMap<KeyType, ValueType>& hashMap, size_t capacity) {
    // Round up to next power of 2 for efficient masking
    size_t actualCapacity = 1;
    while (actualCapacity < capacity) {
        actualCapacity <<= 1;
    }
    
    // Allocate GPU memory for keys and values
    cudaError_t keyResult = cudaMalloc(&hashMap.keys, actualCapacity * sizeof(KeyType));
    if (keyResult != cudaSuccess) return keyResult;
    
    cudaError_t valueResult = cudaMalloc(&hashMap.values, actualCapacity * sizeof(ValueType));
    if (valueResult != cudaSuccess) {
        cudaFree(hashMap.keys);
        return valueResult;
    }
    
    hashMap.capacity = actualCapacity;
    hashMap.mask = actualCapacity - 1;
    hashMap.size = 0;
    
    // Initialize hash map
    int numThreads = 256;
    int numBlocks = (actualCapacity + numThreads - 1) / numThreads;
    initializeGpuHashMap<<<numBlocks, numThreads>>>(hashMap);
    
    return cudaGetLastError();
}

/*! @brief Deallocate GPU hash map memory
 *
 * @param[inout] hashMap    Hash map to deallocate
 */
template<class KeyType, class ValueType>
void deallocateGpuHashMap(GpuHashMap<KeyType, ValueType>& hashMap) {
    if (hashMap.keys) {
        cudaFree(hashMap.keys);
        hashMap.keys = nullptr;
    }
    if (hashMap.values) {
        cudaFree(hashMap.values);
        hashMap.values = nullptr;
    }
    hashMap.capacity = 0;
    hashMap.mask = 0;
    hashMap.size = 0;
}

/*! @brief Batch insert multiple key-value pairs into hash map
 *
 * @param[inout] hashMap       Hash map to insert into
 * @param[in]    keys          Array of keys to insert
 * @param[in]    values        Array of values to insert
 * @param[in]    numElements   Number of elements to insert
 */
template<class KeyType, class ValueType>
__global__ void batchInsertGpuHashMap(GpuHashMap<KeyType, ValueType> hashMap, 
                                     const KeyType* keys, const ValueType* values, size_t numElements) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= numElements) return;
    
    gpuHashMapInsert(hashMap, keys[tid], values[tid]);
}

/*! @brief GPU kernel to map global cluster keys to local halo IDs using hash map
 *
 * @param[in]    globalKeys     Array of global cluster keys
 * @param[out]   localHaloIds   Array to store corresponding local halo IDs
 * @param[in]    numParticles   Number of particles to process
 * @param[in]    hashMap        Hash map containing key-to-ID mappings
 */
template<class ClusterKeyType, class ClusterIdType>
__global__ void mapClusterKeysToHaloIds(
    const ClusterKeyType* globalKeys,
    ClusterIdType* localHaloIds,
    size_t numParticles,
    GpuHashMap<ClusterKeyType, ClusterIdType> hashMap)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= numParticles) return;
    
    ClusterKeyType key = globalKeys[tid];
    ClusterIdType haloId = gpuHashMapLookup(hashMap, key);
    localHaloIds[tid] = haloId;
}

/*! @brief GPU kernel to map global cluster keys to local halo IDs using hash map, including flags
 *
 * @param[in]    globalKeys     Array of global cluster keys
 * @param[out]   localHaloIds   Array to store corresponding local halo IDs
 * @param[out]   flags          Array to store flags for found keys
 * @param[in]    numParticles   Number of particles to process
 * @param[in]    hashMap        Hash map containing key-to-ID mappings
 */
template<class ClusterKeyType, class ClusterIdType>
__global__ void mapClusterKeysToHaloIdsFlagged(
    const ClusterKeyType* globalKeys,
    ClusterIdType* localHaloIds,
    ClusterIdType* flags,
    size_t numParticles,
    GpuHashMap<ClusterKeyType, ClusterIdType> hashMap)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= numParticles) return;
    
    ClusterKeyType key = globalKeys[tid];
    ClusterIdType haloId = gpuHashMapLookup(hashMap, key);
    localHaloIds[tid] = haloId;
    flags[tid] = (haloId == GpuHashMap<ClusterKeyType, ClusterIdType>::EMPTY_VALUE) ? 0 : 1;

}

/*! @brief Host function to create and populate hash map for cluster key mapping
 *
 * @param[in]    clusterKeys      Array of unique cluster keys
 * @param[in]    haloIds          Corresponding local halo IDs
 * @param[in]    numUniqueKeys    Number of unique key-value pairs
 * @param[out]   hashMap          Hash map to populate
 * @return                        cudaError_t indicating success or failure
 */
template<class KeyType, class ValueType>
cudaError_t createClusterKeyHashMap(
    const KeyType* clusterKeys,
    const ValueType* haloIds,
    size_t numUniqueKeys,
    GpuHashMap<KeyType, ValueType>& hashMap)
{
    // Allocate hash map with 1.5x capacity
    size_t capacity = (numUniqueKeys * 3) / 2;
    cudaError_t result = allocateGpuHashMap(hashMap, capacity);
    if (result != cudaSuccess) return result;
    
    // Batch insert into hash map
    int numThreads = 256;
    int numBlocks = (numUniqueKeys + numThreads - 1) / numThreads;
    batchInsertGpuHashMap<<<numBlocks, numThreads>>>(hashMap, clusterKeys, haloIds, numUniqueKeys);
    
    result = cudaGetLastError();
    
    return result;
}

/*! @brief Example function demonstrating complete GPU hash map workflow
 *
 * This class provides a complete interface for managing cluster key to halo ID mappings
 * using GPU hash maps for efficient lookups during parallel cluster processing.
 */
template<class KeyType, class ValueType>
class ClusterKeyHashMapManager {
private:
    GpuHashMap<KeyType, ValueType> hashMap;
    bool initialized;

public:
    ClusterKeyHashMapManager() : initialized(false) {}
    
    ~ClusterKeyHashMapManager() {
        if (initialized) {
            deallocateGpuHashMap(hashMap);
        }
    }
    
    /*! @brief Initialize hash map with cluster key mappings */
    cudaError_t initialize(const KeyType* clusterKeys, 
                           const ValueType* haloIds,
                           const size_t numUniqueKeys) {
        
        cudaError_t result = createClusterKeyHashMap(
            clusterKeys, 
            haloIds, 
            numUniqueKeys,
            hashMap
        );
        
        if (result == cudaSuccess) {
            initialized = true;
        }
        return result;
    }

    /*! @brief Batch lookup of multiple cluster keys */
    cudaError_t lookupBatch(const KeyType* d_keys, ValueType* d_values, size_t numKeys) {
        if (!initialized) {
            throw std::runtime_error("Hash map not initialized");
        }
        int numThreads = 256;
        int numBlocks = (numKeys + numThreads - 1) / numThreads;
        
        mapClusterKeysToHaloIds<<<numBlocks, numThreads>>>(d_keys, d_values, numKeys, hashMap);
        
        return cudaGetLastError();
    }

    /*! @brief Batch lookup of multiple cluster keys, also returns flag for successful retrieval*/
    cudaError_t lookupBatchFlag(const KeyType* d_keys, ValueType* d_values, ValueType* flags, size_t numKeys) {
        if (!initialized) {
            throw std::runtime_error("Hash map not initialized");
        }
        int numThreads = 256;
        int numBlocks = (numKeys + numThreads - 1) / numThreads;
        
        mapClusterKeysToHaloIdsFlagged<<<numBlocks, numThreads>>>(d_keys, d_values, flags, numKeys, hashMap);
        
        return cudaGetLastError();
    }
    
    /*! @brief Get hash map statistics */
    void printStatistics() const {
        if (initialized) {
            printf("Hash Map Statistics:\n");
            printf("  Capacity: %u\n", hashMap.capacity);
            printf("  Size: %u\n", hashMap.size);
            printf("  Load Factor: %.2f\n", (double)hashMap.size / hashMap.capacity);
        }
    }
    
    /*! @brief Check if hash map is initialized */
    bool isInitialized() const { return initialized; }
};

template class ClusterKeyHashMapManager<uint64_t, unsigned>;
template class ClusterKeyHashMapManager<unsigned, unsigned>;
template class ClusterKeyHashMapManager<unsigned long long, unsigned>;

}