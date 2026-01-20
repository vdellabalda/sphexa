/*!
 * @brief Contains the object holding all cluster data
 * @author Vincente Della Balda <vinc.dellabalda@gmail.com>
 */

#pragma once

#include <array>
#include <vector>
#include <variant>

#include "cstone/cuda/cuda_utils.hpp"
#include "cstone/fields/data_util.hpp"
#include "cstone/fields/field_states.hpp"
#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/tree/definitions.h"
#include "cstone/tree/octree.hpp"
#include "cstone/util/reallocate.hpp"

#include "definitions.h"
#include "cluster_data_stubs.hpp"

#if defined(USE_CUDA)
#include "cluster_data_gpu.cuh"
#endif

namespace cluster
{

template<class AccType>
class ClusterData : public cstone::FieldStates<ClusterData<AccType>>
{
public:
    using AcceleratorType = AccType;

    using KeyType   = sph::SphTypes::KeyType;
    using RealType  = sph::SphTypes::CoordinateType;
    using ClusterIdType = cluster::ClusterIdType;
    using ClusterKeyType = cluster::ClusterKeyType;
    using EdgeType = cluster::EdgeType;
    using IdType         = unsigned;

    template<class ValueType>
    using PinnedVec = std::vector<ValueType, PinnedAlloc_t<AcceleratorType, ValueType>>;

    template<class ValueType>
    using FieldVector = std::vector<ValueType, std::allocator<ValueType>>;

    using FieldVariant = std::variant<FieldVector<float>*, FieldVector<double>*, FieldVector<unsigned>*,
                                      FieldVector<uint64_t>*, FieldVector<uint8_t>*>;

    uint64_t iteration{1};
    uint64_t numParticlesGlobal{0};
    uint64_t numParticlesHalos{0};
    IdType   clusterThreshold{32};
    uint64_t numClustersGlobal{0};

    //! @brief default maximum number of edge buffer size per warp before partial DSU is triggered
    unsigned egmax{150};

    //! @brief minimum radius for edge search
    RealType percolationLength{0.0};
    void setPercLength(RealType l) { percolationLength = l; }

    RealType ttot{0.0};

    //! @brief Unified interface to attribute initialization, reading and writing
    template<class Archive>
    void loadOrStoreAttributes(Archive* ar)
    {
        //! @brief load or store an attribute, skips non-existing attributes on load.
        auto optionalIO = [ar](const std::string& attribute, auto* location, size_t attrSize)
        {
            try
            {
                if constexpr (std::is_enum_v<std::decay_t<decltype(*location)>>)
                {
                    // handle pointers to enum by casting to the underlying type
                    using EType = std::decay_t<decltype(*location)>;
                    using UType = std::underlying_type_t<EType>;
                    auto tmp    = static_cast<UType>(*location);
                    ar->stepAttribute(attribute, &tmp, attrSize);
                    *location = static_cast<EType>(tmp);
                }
                else { ar->stepAttribute(attribute, location, attrSize); }
            }
            catch (std::out_of_range&)
            {
                if (ar->rank() == 0)
                {
                    std::cout << "Attribute " << attribute << " not set in file, setting to default value " << *location
                              << std::endl;
                }
            }
        };

        ar->stepAttribute("iteration", &iteration, 1);
        ar->stepAttribute("numParticlesGlobal", &numParticlesGlobal, 1);
        optionalIO("percolationLength", &percolationLength, 1);
        optionalIO("egmax", &egmax, 1);
        ar->stepAttribute("time", &ttot, 1);
    }

    //! @brief non-stateful variables for statistics
    uint64_t totalNeighbors{0};

    /*! @brief Particle fields
     *
     * The length of these arrays equals the local number of particles including halos
     * if the field is active and is zero if the field is inactive.
     */
         
    FieldVector<IdType>             halo_id;
    FieldVector<IdType>             flagged;
    FieldVector<ClusterKeyType>     localClusterKeys;
    FieldVector<ClusterKeyType>     globalClusterKeys;
    
    // temporary arrays for sorting and run-length encoding
    FieldVector<ClusterIdType>      idBuf;
    FieldVector<ClusterKeyType>     keyBuf;
    FieldVector<IdType>             thresholdMask;
    FieldVector<ClusterIdType>      idMap;

    // temporary array for edge construction
    FieldVector<ClusterKeyType>     edgeSrc;
    FieldVector<ClusterKeyType>     edgeDst;
    FieldVector<EdgeType>           edges;

    /*! @brief Cluster fields */
    FieldVector<ClusterKeyType>     uniqueKeys;
    FieldVector<IdType>             keyCounts;
    FieldVector<IdType>             clusterParents;
    FieldVector<IdType>             clusterSizes;
    FieldVector<unsigned>           clusterOwner;
    FieldVector<ClusterKeyType>     nonLocalKeys;
    FieldVector<ClusterKeyType>     localKeys;

    // Number of local clusters
    FieldVector<IdType>             numClusters;

    DeviceClusterData_t<AccType> devData;

    /*! @brief
     * Name of each field as string for use e.g in HDF5 output. Order has to correspond to what's returned by data().
     */
    inline static constexpr std::array fieldNames{
        "halo_id", "flagged", "idBuf", "keyBuf", "localClusterKeys", "globalClusterKeys"};

    //! @brief dataset prefix to be prepended to fieldNames for structured output
    static const inline std::string prefix{};

    static_assert(!cstone::HaveGpu<AcceleratorType>{} || fieldNames.size() == DeviceClusterData_t<AccType>::fieldNames.size(),
                  "ClusterData on CPU and GPU must have the same fields");

    /*! @brief return a tuple of field references
     *
     * Note: this needs to be in the same order as listed in fieldNames
     */
    auto dataTuple()
    {
        auto ret = std::tie(halo_id, flagged, idBuf, keyBuf, localClusterKeys, globalClusterKeys);

#if defined(__clang__) || __GNUC__ > 11
        static_assert(std::tuple_size_v<decltype(ret)> == fieldNames.size());
#endif
        return ret;
    }

    /*! @brief return a vector of pointers to field vectors
     *
     * We implement this by returning an rvalue to prevent having to store pointers and avoid
     * non-trivial copy/move constructors.
     */
    auto data()
    {
        return std::apply([](auto&... fields) { return std::array<FieldVariant, sizeof...(fields)>{&fields...}; },
                          dataTuple());
    }

    /*! @brief mark fields file output
     *
     * @param outFields  list of field names
     *
     * Selected fields that match existing names contained in @a fieldNames will be removed from the argument
     * @p field names.
     */
    void setOutputFields(std::vector<std::string>& outFields)
    {
        auto hasField = [](const std::string& field)
        { return cstone::getFieldIndex(field, fieldNames) < fieldNames.size(); };

        std::copy_if(outFields.begin(), outFields.end(), std::back_inserter(outputFieldNames), hasField);
        outputFieldIndices = cstone::fieldStringsToInt(outputFieldNames, fieldNames);
        std::for_each(outputFieldNames.begin(), outputFieldNames.end(), [](auto& f) { f = prefix + f; });

        outFields.erase(std::remove_if(outFields.begin(), outFields.end(), hasField), outFields.end());
    }

    void resize(size_t size)
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
                std::visit([size, gr = allocGrowthRate_](auto* arg) { reallocate(*arg, size, gr); }, data_[i]);
            }
        }

        devData.resize(size, allocGrowthRate_);
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

    //! @brief resize GPU arrays if in use, CPU arrays otherwise
    void resizeAcc(size_t size)
    {
        if (cstone::HaveGpu<AccType>{}) { devData.resize(size, allocGrowthRate_); }
        else { resize(size); }
    }



    //! @brief return the size of GPU arrays if in use, CPU arrays otherwise
    size_t accSize()
    {
        if (cstone::HaveGpu<AccType>{}) { return devData.size(); }
        else { return size(); }
    }

    //! @brief cluster fields selected for file output
    std::vector<int>         outputFieldIndices;
    std::vector<std::string> outputFieldNames;

    float getAllocGrowthRate() const { return allocGrowthRate_; }
    RealType getPercLength() const { return percolationLength; }
    uint64_t getNumParticlesHalos() const { return numParticlesHalos; }
    IdType getClusterThreshold() const {return clusterThreshold; }

    IdType getNumClusters()
    {
        if (cstone::HaveGpu<AccType>{})
        {
            memcpyD2H(rawPtr(devData.numClusters), 1, rawPtr(numClusters));
            return numClusters.empty() ? 0 : numClusters[0];
        }
        else
        {
            return numClusters.empty() ? 0 : numClusters[0];
        }
    };

private:

    //! @brief buffer growth factor when reallocating
    float allocGrowthRate_{1.05};
};

} // namespace sphexa
