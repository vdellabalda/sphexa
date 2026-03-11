/*!
 * @brief Contains the object holding all cluster data
 * @author Vincente Della Balda <vinc.dellabalda@gmail.com>
 */

#pragma once

#include <array>
#include <iostream>
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

namespace cluster
{

template<class AccType>
class ClusterData : public cstone::FieldStates<ClusterData<AccType>>
{
public:
    using AcceleratorType = AccType;

    using KeyType   = sph::SphTypes::KeyType;
    using RealType  = sph::SphTypes::CoordinateType;
    using HydroType = sph::SphTypes::HydroType;
    using MassType  = sph::SphTypes::Tmass;
    using ClusterIdType = cluster::ClusterIdType;
    using ClusterKeyType = cluster::ClusterKeyType;
    using EdgeType = cluster::EdgeType;
    using IdType         = uint32_t;

    template<class ValueType>
    using FieldVector =
        std::conditional_t<cstone::HaveGpu<AccType>{}, cstone::DeviceVector<ValueType>, std::vector<ValueType>>;

    using FieldVariant = std::variant<FieldVector<float>*, FieldVector<double>*, FieldVector<unsigned>*,
                                      FieldVector<uint64_t>*, FieldVector<uint8_t>*>;

    uint64_t iteration{1};
    uint32_t numParticles{0};               // local number of particles
    uint64_t numParticlesHalos{0};          // local number of particles including halos
    uint64_t numClustersGlobal{0};          // global number of clusters
    uint64_t numClustersNonLocal{0};        // number of non-local clusters
    bool     firstIter{true};               // flag to indicate first iteration of clustering
    
    //! @brief minimum radius for edge search
    RealType percolationLength{0.0};
    void setPercLength(RealType l) { percolationLength = l; }

    //! @brief merge factor for density zones in FOF group
    RealType mergeFactor{0.3};
    void setMergeFactor(RealType f) { mergeFactor = f; }

    //! @brief minimum number of particles per cluster
    unsigned clusterThreshold{32};
    void setThreshold(unsigned t) { clusterThreshold = t; }

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
        optionalIO("percolationLength", &percolationLength, 1);
        ar->stepAttribute("time", &ttot, 1);
    }

    /*! @brief Particle fields
     *
     * The length of these arrays equals the local number of particles including halos
     * if the field is active and is zero if the field is inactive.
     */
         
    FieldVector<ClusterIdType>      halo_id; // FOF cluster ID
    FieldVector<unsigned>           flagged; // temporary field for marking particles during clustering
    FieldVector<ClusterIdType>      localClusterIds; // temporary field for cluster keys of local particles
    FieldVector<ClusterKeyType>     globalClusterKeys; // temporary field for cluster keys of non-local particles
    FieldVector<HydroType>          hTight;       // Smoothing length for subcluster detection
    
    FieldVector<ClusterIdType>      sub_id;  // used for subcluster ID when subclustering is enabled

    // Possibly exchange these with scratch space from unused fields
    FieldVector<ClusterIdType>      idBuf;  // buffer array for ClusterIdType
    FieldVector<ClusterKeyType>     keyBuf; // buffer array for ClusterKeyType
    FieldVector<ClusterKeyType>     scratchBuf; // buffer array for any type

    // 
    // temporary array for edge construction
    FieldVector<EdgeType>           edges;   // only used in CPU version
    FieldVector<ClusterKeyType>     edgeSrc; // only used in CPU version
    FieldVector<ClusterKeyType>     edgeDst; // only used in CPU version

    /*! @brief Cluster fields */
    FieldVector<ClusterKeyType>     nonLocalKeys;
    FieldVector<ClusterKeyType>     uniqueKeys; // only used in CPU version
    FieldVector<IdType>             localKeyCounts; // only used in CPU version
    FieldVector<IdType>             clusterParents; // only used in CPU version
    FieldVector<IdType>             clusterSizes; // only used in CPU version

    //DeviceClusterData_t<AccType> devData;

    //! @brief non-stateful variables for statistics
    size_t stackUsedEc;
    size_t subStackUsedEc;
    size_t edgesFoundEc;
    size_t subEdgesFoundEc;

    /*! @brief
     * Name of each field as string for use e.g in HDF5 output. Order has to correspond to what's returned by data().
     */
    inline static constexpr std::array fieldNames{
        "halo_id", "sub_id", "flagged", "localClusterIds", "globalClusterKeys", "hTight", "idBuf", "keyBuf", "scratchBuf"};

    //! @brief dataset prefix to be prepended to fieldNames for structured output
    static const inline std::string prefix{};

    /*! @brief return a tuple of field references
     *
     * Note: this needs to be in the same order as listed in fieldNames
     */
    auto dataTuple()
    {
        auto ret = std::tie(halo_id, sub_id, flagged, localClusterIds, globalClusterKeys, hTight, idBuf, keyBuf, scratchBuf);

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
            using VecType = std::decay_t<decltype(*devVectorPtr)>;
            if (devVectorPtr->capacity() < size) { *devVectorPtr = VecType{}; }
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

    util::array<std::size_t, 5> memStats()
    {
        auto        data_ = data();
        std::size_t sumOfSize{0}, sumOfCap{0};
        for (size_t i = 0; i < data_.size(); ++i)
        {
            sumOfSize +=
                std::visit([]<class V>(V* arg) { return sizeof(typename V::value_type) * arg->size(); }, data_[i]);
            sumOfCap +=
                std::visit([]<class V>(V* arg) { return sizeof(typename V::value_type) * arg->capacity(); }, data_[i]);
        }

        std::size_t free{0}, total{0};
        return {size(), sumOfSize, sumOfCap, free, total};
    }

    //! @brief cluster fields selected for file output
    std::vector<int>         outputFieldIndices;
    std::vector<std::string> outputFieldNames;

    float getAllocGrowthRate() const { return allocGrowthRate_; }
    RealType getPercLength() const { return percolationLength; }
    uint64_t getNumParticlesHalos() const { return numParticlesHalos; }
    IdType getClusterThreshold() const {return clusterThreshold; }

private:

    //! @brief buffer growth factor when reallocating
    float allocGrowthRate_{1.05};
};

} // namespace cluster
