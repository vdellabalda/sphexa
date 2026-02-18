/*!
 * @brief HDF5 output data container for cluster-sorted particle data
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
#include "cstone/fields/field_get.hpp"
#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/util/reallocate.hpp"

#include "sph/types.hpp"
#include "definitions.h"
#include "hdf5_data_utils.hpp"
#include "halo_data.hpp"

namespace cluster
{

template<class AccType>
class HDF5Data : public cstone::FieldStates<HDF5Data<AccType>>
{
public:
    using AcceleratorType = AccType;
    using RealType = sph::SphTypes::CoordinateType;
    using HydroType = sph::SphTypes::HydroType;
    using Tmass = sph::SphTypes::Tmass;
    using IdType = uint32_t;

    template<class ValueType>
    using FieldVector =
        std::conditional_t<cstone::HaveGpu<AccType>{}, cstone::DeviceVector<ValueType>, std::vector<ValueType>>;

    using FieldVariant = std::variant<FieldVector<double>*, FieldVector<float>*, FieldVector<unsigned>*, FieldVector<uint64_t>*>;

    // Particle data fields sorted by cluster
    FieldVector<RealType> x, y, z;
    FieldVector<HydroType> vx, vy, vz;
    FieldVector<Tmass> m;
    FieldVector<IdType> halo_id;
    FieldVector<uint64_t> id;

    // Field names for proper allocation management
    inline static constexpr std::array fieldNames{"x", "y", "z", "vx", "vy", "vz", "m", "halo_id", "id"};
    static const inline std::string prefix{"hdf5_"};

    // Return tuple of field references (same order as fieldNames)
    auto dataTuple()
    {
        auto ret = std::tie(x, y, z, vx, vy, vz, m, halo_id, id);
#if defined(__clang__) || __GNUC__ > 11
        static_assert(std::tuple_size_v<decltype(ret)> == fieldNames.size());
#endif
        return ret;
    }

    // Return array of field pointers for allocation management
    auto data()
    {
        return std::apply([](auto&... fields) { 
            return std::array<FieldVariant, sizeof...(fields)>{&fields...}; 
        }, dataTuple());
    }

float allocGrowthRate_ = 1.01f;

void resize(size_t size)
    {
        auto data_ = data();

        // Deallocate vectors that are too small (following parent class pattern)
        auto deallocateVector = [size](auto* devVectorPtr)
        {
            using VecType = std::decay_t<decltype(*devVectorPtr)>;
            if (devVectorPtr->capacity() < size) { *devVectorPtr = VecType{}; }
        };

        for (size_t i = 0; i < data_.size(); ++i)
        {
            if (this->isAllocated(i) && not this->isConserved(i)) { 
                std::visit(deallocateVector, data_[i]); 
            }
        }

        // Reallocate all active fields
        for (size_t i = 0; i < data_.size(); ++i)
        {
            if (this->isAllocated(i))
            {
                std::visit([size, gr = allocGrowthRate_](auto* arg) { 
                    reallocate(*arg, size, gr); 
                }, data_[i]);
            }
        }
    }

    size_t size() const
    {
        auto data_ = const_cast<HDF5Data*>(this)->data();
        for (size_t i = 0; i < data_.size(); ++i)
        {
            if (this->isAllocated(i))
            {
                return std::visit([](auto* arg) { return arg->size(); }, data_[i]);
            }
        }
        return 0;
    }

    // Activate all fields for allocation
    void activateAllFields()
    {
        this->setConserved("x", "y", "z", "vx", "vy", "vz", "m", "halo_id", "id");
    }

    // Activate only specific fields
    void activateFields(const std::vector<std::string>& fields)
    {
        auto hasField = [](const std::string& field)
        { return cstone::getFieldIndex(field, fieldNames) < fieldNames.size(); };
        
        for (const auto& field : fields) {
            if (hasField(field)) {
                this->setConserved(field);
            }
        }
    }
    
    HDF5HostData getHostData(std::vector<std::string>& fields) const
    {   
        HDF5HostData hostData;
        // Only copy requested fields
        auto hasField = [&fields](const std::string& field) {
            return std::find(fields.begin(), fields.end(), field) != fields.end();
        };

        hostData.x = hasField("x") ? toHost(x) : std::vector<double>{};
        hostData.y = hasField("y") ? toHost(y) : std::vector<double>{};
        hostData.z = hasField("z") ? toHost(z) : std::vector<double>{};
        hostData.vx = hasField("vx") ? toHost(vx) : std::vector<float>{};
        hostData.vy = hasField("vy") ? toHost(vy) : std::vector<float>{};
        hostData.vz = hasField("vz") ? toHost(vz) : std::vector<float>{};
        hostData.m = hasField("m") ? toHost(m) : std::vector<float>{};
        hostData.halo_id = hasField("halo_id") ? toHost(halo_id) : std::vector<unsigned>{};
        hostData.id = hasField("id") ? toHost(id) : std::vector<uint64_t>{};


        return hostData;
    }

    std::vector<ClusterInfo> createClusterInfos(const cluster::HaloData<AccType>& haloData) {
        std::vector<ClusterInfo> clusterInfos(haloData.numClustersGlobal);
        auto uniqueId = toHost(haloData.id);
        auto localCount = toHost(haloData.localSize);
        auto globalCount = toHost(haloData.globalSize);
        auto localOffset = toHost(haloData.localOffset);
        auto globalOffset = toHost(haloData.globalOffset);
        
        for (size_t i = 0; i < haloData.numClustersGlobal; ++i) {
            ClusterInfo info;
            info.clusterId    = uniqueId[i];
            info.localCount   = localCount[i];
            info.globalCount  = globalCount[i];
            info.localOffset   = localOffset[i];
            info.globalOffset = globalOffset[i];
            info.haloOffset   = computeHaloOffset(localCount[i]);
            clusterInfos[i] = info;
        }
        return clusterInfos;
    }

    // Generic gather function for multiple fields by name
    template<class SimulationDataType>
    void gatherFields(const std::vector<std::string>& fieldNames, 
                     const SimulationDataType& simData, 
                     const FieldVector<IdType>& gatherMap)
    {   
        auto& d = simData.hydro;
        auto& c = simData.clust;

        size_t gatheredSize = gatherMap.size();
        resize(gatheredSize);
        
        // Gather each requested field
        for (const auto& fieldName : fieldNames) {
            gatherSingleField(fieldName, d, c, gatherMap);
        }
    }

private:
    // Gather a single field by name using runtime dispatch
    template<class ParticleDataset, class ClusterDataset>
    void gatherSingleField(const std::string& fieldName, 
                          ParticleDataset& particleDataset,
                          ClusterDataset& clusterDataset,
                          const FieldVector<IdType>& gatherMap)
    {   
        auto fieldIndex = cstone::getFieldIndex(fieldName, fieldNames);
        if (fieldIndex >= fieldNames.size()) {
            printf("Warning: Requested field '%s' not found in particle dataset. Skipping.\n", fieldName.c_str());
            return;
        }
        constexpr bool useGpu = cstone::HaveGpu<typename ParticleDataset::AcceleratorType>{};
        auto gatherSpan = std::span<const IdType>(gatherMap.data(), gatherMap.size());

        // Gather from particle dataset to HDF5Data field
        if (fieldName == "x") {
            cstone::gatherAcc<useGpu>(gatherSpan, rawPtr(particleDataset.x), rawPtr(x));
        }
        else if (fieldName == "y") {
            cstone::gatherAcc<useGpu>(gatherSpan, rawPtr(particleDataset.y), rawPtr(y));
        }
        else if (fieldName == "z") {
            cstone::gatherAcc<useGpu>(gatherSpan, rawPtr(particleDataset.z), rawPtr(z));
        }
        else if (fieldName == "vx") {
            cstone::gatherAcc<useGpu>(gatherSpan, rawPtr(particleDataset.vx), rawPtr(vx));
        }
        else if (fieldName == "vy") {
            cstone::gatherAcc<useGpu>(gatherSpan, rawPtr(particleDataset.vy), rawPtr(vy));
        }
        else if (fieldName == "vz") {
            cstone::gatherAcc<useGpu>(gatherSpan, rawPtr(particleDataset.vz), rawPtr(vz));
        }
        else if (fieldName == "m") {
            cstone::gatherAcc<useGpu>(gatherSpan, rawPtr(particleDataset.m), rawPtr(m));
        }
        else if (fieldName == "halo_id") {
            cstone::gatherAcc<useGpu>(gatherSpan, rawPtr(clusterDataset.halo_id), rawPtr(halo_id));
        }
        else if (fieldName == "id") {
            cstone::gatherAcc<useGpu>(gatherSpan, rawPtr(particleDataset.id), rawPtr(id));
        }
        
    }

    float getAllocGrowthRate() const { return allocGrowthRate_; }

};

} // namespace cluster