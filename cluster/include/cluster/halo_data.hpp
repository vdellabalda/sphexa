/*!
 * @brief Contains the object holding all halo data
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
class HaloData : public cstone::FieldStates<HaloData<AccType>>
{
public:
    using AcceleratorType = AccType;

    using RealType  = sph::SphTypes::CoordinateType;
    using IdType         = uint32_t;
    using HydroType = sph::SphTypes::HydroType;
    using Tmass     = sph::SphTypes::Tmass;

    template<class ValueType>
    using FieldVector =
        std::conditional_t<cstone::HaveGpu<AccType>{}, cstone::DeviceVector<ValueType>, std::vector<ValueType>>;

    using FieldVariant = std::variant<FieldVector<float>*, FieldVector<double>*, FieldVector<unsigned>*,
                                      FieldVector<uint64_t>*, FieldVector<uint8_t>*>;

    uint64_t iteration{1};
    uint32_t numClustersLocal{0};
    uint32_t numClustersGlobal{1};
    size_t nClusteredLocal{0};
    size_t nClusteredGlobal{0};
    IdType   clusterThreshold{32};

    RealType percolationLength{0.0};
    void setPercLength(RealType l) { percolationLength = l; }

    /*! @brief Halo fields
     *
     * The length of these arrays equals the global number of halos
     * if the field is active and is zero if the field is inactive.
     */
    FieldVector<IdType>             cId;
    FieldVector<Tmass>              cMass;
    FieldVector<Tmass>              xCenter;
    FieldVector<Tmass>              yCenter;
    FieldVector<Tmass>              zCenter;
    FieldVector<HydroType>          xVelocity;
    FieldVector<HydroType>          yVelocity;
    FieldVector<HydroType>          zVelocity;
    FieldVector<uint32_t>           localSize;
    FieldVector<uint32_t>           globalSize;
    FieldVector<uint32_t>           localOffset;
    FieldVector<uint32_t>           globalOffset;
    
    // Number of local clusters
    FieldVector<IdType>             numHalos;
    
    // Gather map from full particle set to sorted and grouped halo particle set
    FieldVector<IdType>             particleToHaloMap;

    /*! @brief
     * Name of each field as string for use e.g in HDF5 output. Order has to correspond to what's returned by data().
     */
    inline static constexpr std::array fieldNames{
        "cId", "globalSize", "cMass", "xCenter", "yCenter", "zCenter", "xVelocity", "yVelocity", "zVelocity",
        "localSize", "localOffset", "globalOffset"};
    
    //! @brief dataset prefix to be prepended to fieldNames for structured output
    static const inline std::string prefix{};

    /*! @brief return a tuple of field references
     *
     * Note: this needs to be in the same order as listed in fieldNames
     */
    auto dataTuple()
    {
        auto ret = std::tie(
            cId, globalSize, cMass, xCenter, yCenter, zCenter, xVelocity, yVelocity, zVelocity,
            localSize, localOffset, globalOffset
        );

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

    //! @brief halo fields selected for file output
    std::vector<int>         outputFieldIndices;
    std::vector<std::string> outputFieldNames;

    float getAllocGrowthRate() const { return allocGrowthRate_; }
    RealType getPercLength() const { return percolationLength; }
    IdType getClusterThreshold() const {return clusterThreshold; }
    uint32_t getNumClustersGlobal() const { return numClustersGlobal; }
    uint32_t getNumClustersLocal() const { return numClustersLocal; }

private:

    //! @brief buffer growth factor when reallocating
    float allocGrowthRate_{1.05};
};

} // namespace cluster