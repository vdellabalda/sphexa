/*! @file
 * @brief Switch and stub classes for ClusterData to abstract and manage GPU device acceleration behavior
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 * @author Vincente Della Balda <vinc.dellabalda@gmail.com>
 * 
 */

#pragma once

#include "cstone/primitives/primitives_acc.hpp"

template<class T>
class pinned_allocator;

namespace cluster
{

//! @brief std::allocator on the CPU, pinned_allocator on the GPU
template<class Accelerator, class T>
using PinnedAlloc_t = std::conditional_t<cstone::HaveGpu<Accelerator>{}, pinned_allocator<T>, std::allocator<T>>;

//! @brief stub for use in CPU code
struct DeviceClusterDataFacade
{
    void   resize(size_t, float) {}
    size_t size() { return 0; }

    template<class... Ts>
    void setConserved(Ts...)
    {
    }

    template<class... Ts>
    void setDependent(Ts...)
    {
    }

    template<class... Ts>
    void release(Ts...)
    {
    }

    template<class... Ts>
    void acquire(Ts...)
    {
    }

    template<class Table>
    void uploadTables(const Table&, const Table&)
    {
    }

    inline static constexpr std::array fieldNames{0};
};

class DeviceClusterData;

//! @brief Just a facade on the CPU, DeviceClusterData on the GPU
template<class Accelerator>
using DeviceClusterData_t = std::conditional_t<cstone::HaveGpu<Accelerator>{}, DeviceClusterData, DeviceClusterDataFacade>;

} // namespace sphexa