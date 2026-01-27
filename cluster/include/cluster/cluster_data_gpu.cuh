/*! @file
 * @brief Contains the object holding cluster data on the GPU
 * @author Vincente Della Balda <vinc.dellabalda@gmail.com>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <variant>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/cuda/device_vector.h"
#include "cstone/fields/field_states.hpp"
#include "cstone/primitives/primitives_gpu.h"
#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/tree/definitions.h"
#include "cstone/util/reallocate.hpp"

#include "definitions.h"

#include "sph/types.hpp"

namespace cluster
{

class DeviceClusterData : public cstone::FieldStates<DeviceClusterData>
{
    template<class FType>
    using DevVector = cstone::DeviceVector<FType>;

    using KeyType   = sph::SphTypes::KeyType;
    using RealType  = sph::SphTypes::CoordinateType;
    using ClusterIdType = cluster::ClusterIdType;
    using ClusterKeyType = cluster::ClusterKeyType;
    using EdgeType = cluster::EdgeType;
    using IdType         = unsigned;

public:
    // number of CUDA streams to use
    static constexpr int NST = 2;

    struct neighbors_stream
    {
        cudaStream_t stream;
    };

    struct neighbors_stream d_stream[NST];

    /*! @brief Particle fields
     *
     * The length of these arrays equals the local number of particles including halos
     * if the field is active and is zero if the field is inactive.
     */
    DevVector<ClusterIdType>    halo_id;
    DevVector<ClusterIdType>    flagged;
    DevVector<ClusterKeyType>   localClusterKeys;
    DevVector<ClusterKeyType>   globalClusterKeys;

    // temporary arrays for sorting and run-length encoding
    DevVector<ClusterIdType>    idBuf;
    DevVector<ClusterKeyType>   keyBuf;
    DevVector<unsigned>         thresholdMask;
    DevVector<ClusterIdType>    idMap;
    
    // temporary array for edge construction
    DevVector<ClusterKeyType>   edgeSrc;
    DevVector<ClusterKeyType>   edgeDst;
    DevVector<EdgeType>         edges;

    DevVector<cstone::LocalIndex> traversalStack;

    /*! @brief Cluster fields */
    DevVector<ClusterKeyType>   uniqueKeys;
    DevVector<IdType>           keyCounts;
    DevVector<IdType>           clusterParents;
    DevVector<unsigned>         clusterOwner;
    DevVector<ClusterKeyType>   nonLocalKeys;
    DevVector<ClusterIdType>    nonLocalIds;
    DevVector<ClusterKeyType>   localKeys;

    // Number of local clusters
    DevVector<unsigned>           numClusters;
    

    //! @brief non-stateful variables for statistics
    size_t stackUsedEc;
    size_t edgesFoundEc;

    /*! @brief
     * Name of each field as string for use e.g in HDF5 output. Order has to correspond to what's returned by data().
     */
    inline static constexpr std::array fieldNames{"halo_id", "flagged", "idBuf", "keyBuf", "localClusterKeys", "globalClusterKeys"};

    /*! @brief return a tuple of field references
     *
     * Note: this needs to be in the same order as listed in fieldNames
     */
    auto dataTuple()
    {
        auto ret = std::tie(halo_id, flagged, idBuf, keyBuf, localClusterKeys, globalClusterKeys);

        static_assert(std::tuple_size_v<decltype(ret)> == fieldNames.size());
        return ret;
    }

    /*! @brief return a vector of pointers to field vectors
     *
     * We implement this by returning an rvalue to prevent having to store pointers and avoid
     * non-trivial copy/move constructors.
     */
    auto data()
    {
        using FieldType = std::variant<DevVector<float>*, DevVector<double>*, DevVector<unsigned>*,
                                       DevVector<uint64_t>*, DevVector<uint8_t>*>;

        return std::apply([](auto&... fields) { return std::array<FieldType, sizeof...(fields)>{&fields...}; },
                          dataTuple());
    }

    void resize(size_t size, float growthRate)
    {
        auto data_ = data();

        auto deallocateVector = [size](auto* devVectorPtr)
        {
            using DevVector = std::decay_t<decltype(*devVectorPtr)>;
            if (devVectorPtr->capacity() < size) { *devVectorPtr = DevVector{}; }
        };

        for (size_t i = 0; i < data_.size(); ++i)
        {
            if (this->isAllocated(i) && not this->isConserved(i)) { std::visit(deallocateVector, data_[i]); }
        }

        for (size_t i = 0; i < data_.size(); ++i)
        {
            if (this->isAllocated(i))
            {
                std::visit([size, growthRate](auto* arg) { reallocate(*arg, size, growthRate); }, data_[i]);
            }
        }
    }

    size_t size()
    {
        auto data_ = data();
        for (size_t i = 0; i < data_.size(); ++i)
        {
            if (this->isAllocated(i))
            {
                return std::visit([](auto* arg) { return arg->size(); }, data_[i]);
            }
        }
        return 0;
    }

    DeviceClusterData()
    {
        for (int i = 0; i < NST; ++i)
        {
            checkGpuErrors(cudaStreamCreate(&d_stream[i].stream));
        }
    }

    ~DeviceClusterData()
    {
        for (int i = 0; i < NST; ++i)
        {
            checkGpuErrors(cudaStreamDestroy(d_stream[i].stream));
        }
    }
};

} // namespace sphexa
