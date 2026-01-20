/*! @file
 * @brief Contains the object holding halo data on the GPU
 * @author Vincente Della Balda <vinc.dellabalda@gmail.com>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <variant>

#include "cstone/cuda/cuda_utils.hpp"
#include "cstone/fields/data_util.hpp"
#include "cstone/fields/field_states.hpp"
#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/tree/definitions.h"
#include "cstone/tree/octree.hpp"
#include "cstone/util/reallocate.hpp"
#include "cstone/util/array.hpp"


#include "definitions.h"

#include "sph/types.hpp"

namespace halo
{

class DeviceHaloData : public cstone::FieldStates<DeviceHaloData>
{
    template<class FType>
    using DevVector = cstone::DeviceVector<FType>;

    using RealType  = sph::SphTypes::CoordinateType;
    using HydroType = sph::SphTypes::HydroType;
    using Tmass     = sph::SphTypes::Tmass;

    using IdType    = unsigned;

public:
    // number of CUDA streams to use
    static constexpr int NST = 2;

    struct neighbors_stream
    {
        cudaStream_t stream;
    };

    struct neighbors_stream d_stream[NST];

    /*! @brief Halo fields
     *
     * The length of these arrays equals the local number of halos
     * if the field is active and is zero if the field is inactive.
     */

    DevVector<IdType>             globalId;
    DevVector<IdType>             localId;
    DevVector<Tmass>              mass;
    DevVector<Tmass>              centerX;
    DevVector<Tmass>              centerY;
    DevVector<Tmass>              centerZ;
    DevVector<HydroType>          velocityX;
    DevVector<HydroType>          velocityY;
    DevVector<HydroType>          velocityZ;
    DevVector<IdType>             sizes;
    DevVector<IdType>             ownership; // rank owning the halo

    DevVector<util::array<unsigned, 2>> keyOwnerPair;

    /*! @brief
     * Name of each field as string for use e.g in HDF5 output. Order has to correspond to what's returned by data().
     */
    inline static constexpr std::array fieldNames{
        "globalId",
        "localId",
        "mass",
        "centerX",
        "centerY",
        "centerZ",
        "velocityX",
        "velocityY",
        "velocityZ",
        "sizes",
        "ownership"};

    /*! @brief return a tuple of field references
     *
     * Note: this needs to be in the same order as listed in fieldNames
     */
    auto dataTuple()
    {
        auto ret = std::tie(globalId, localId, mass, centerX, centerY, centerZ,
                            velocityX, velocityY, velocityZ, sizes, ownership);

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
        using FieldType = std::variant<DevVector<float>*, DevVector<double>*, DevVector<unsigned>*>;

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

    DeviceHaloData()
    {
        for (int i = 0; i < NST; ++i)
        {
            checkGpuErrors(cudaStreamCreate(&d_stream[i].stream));
        }
    }

    ~DeviceHaloData()
    {
        for (int i = 0; i < NST; ++i)
        {
            checkGpuErrors(cudaStreamDestroy(d_stream[i].stream));
        }
    }
};

} // namespace cluster