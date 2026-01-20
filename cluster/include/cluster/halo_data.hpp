/*!
 * @brief Contains the object holding all halo data
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
#include "halo_data_stubs.hpp"

#if defined(USE_CUDA)
#include "halo_data_gpu.cuh"
#endif

namespace halo
{

template<class AccType>
class HaloData : public cstone::FieldStates<HaloData<AccType>>
{
public:
    using AcceleratorType = AccType;

    using RealType  = sph::SphTypes::CoordinateType;
    using IdType         = unsigned;
    using HydroType = sph::SphTypes::HydroType;
    using Tmass     = sph::SphTypes::Tmass;

    template<class ValueType>
    using PinnedVec = std::vector<ValueType, PinnedAlloc_t<AcceleratorType, ValueType>>;

    template<class ValueType>
    using FieldVector = std::vector<ValueType, std::allocator<ValueType>>;

    using FieldVariant = std::variant<FieldVector<float>*, FieldVector<double>*, FieldVector<unsigned>*,
                                      FieldVector<uint64_t>*, FieldVector<uint8_t>*>;

    uint64_t iteration{1};
    uint64_t numHalosLocal{0};
    uint64_t numHalosGlobal{0};
    IdType   clusterThreshold{32};

    RealType percolationLength{0.0};
    void setPercLength(RealType l) { percolationLength = l; }

    /*! @brief Halo fields
     *
     * The length of these arrays equals the local number of halos
     * if the field is active and is zero if the field is inactive.
     */

    FieldVector<IdType>             globalId;
    FieldVector<IdType>             localId;
    FieldVector<Tmass>              mass;
    FieldVector<Tmass>              centerX;
    FieldVector<Tmass>              centerY;
    FieldVector<Tmass>              centerZ;
    FieldVector<HydroType>          velocityX;
    FieldVector<HydroType>          velocityY;
    FieldVector<HydroType>          velocityZ;
    FieldVector<IdType>             sizes;
    FieldVector<IdType>             ownership; // rank owning the halo

    FieldVector<util::array<unsigned, 2>> keyOwnerPair;

    DeviceHaloData_t<AccType> devData;

    /*! @brief
     * Name of each field as string for use e.g in HDF5 output. Order has to correspond to what's returned by data().
     */
    inline static constexpr std::array fieldNames{
        "globalId", "localId", "mass", "centerX", "centerY", "centerZ",
        "velocityX", "velocityY", "velocityZ", "sizes", "ownership"};
    
    //! @brief dataset prefix to be prepended to fieldNames for structured output
    static const inline std::string prefix{};

    static_assert(!cstone::HaveGpu<AcceleratorType>{} || fieldNames.size() == DeviceHaloData_t<AccType>::fieldNames.size(),
                  "HaloData on CPU and GPU must have the same fields");

    /*! @brief return a tuple of field references
     *
     * Note: this needs to be in the same order as listed in fieldNames
     */
    auto dataTuple()
    {
        auto ret = std::tie(globalId, localId, mass, centerX, centerY, centerZ,
                            velocityX, velocityY, velocityZ, sizes, ownership);

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
    void setOutputFields()
    {
        auto hasField = [](const std::string& field)
        { return cstone::getFieldIndex(field, fieldNames) < fieldNames.size(); };

        std::copy_if(fieldNames.begin(), fieldNames.end(), std::back_inserter(outputFieldNames), hasField);
        outputFieldIndices = cstone::fieldStringsToInt(outputFieldNames, fieldNames);
        std::for_each(outputFieldNames.begin(), outputFieldNames.end(), [](auto& f) { f = prefix + f; });
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

    //! @brief halo fields selected for file output
    std::vector<int>         outputFieldIndices;
    std::vector<std::string> outputFieldNames;

    float getAllocGrowthRate() const { return allocGrowthRate_; }
    RealType getPercLength() const { return percolationLength; }
    IdType getClusterThreshold() const {return clusterThreshold; }
    uint64_t getNumHalosGlobal() const { return numHalosGlobal; }
    uint64_t getNumHalosLocal() const { return numHalosLocal; }

private:

    //! @brief buffer growth factor when reallocating
    float allocGrowthRate_{1.05};
};

} // namespace cluster