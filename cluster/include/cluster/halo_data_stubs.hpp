/*! @file
 * @brief Switch and stub classes for HaloData to abstract and manage GPU device acceleration behavior
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 * @author Vincente Della Balda <vinc.dellabalda@gmail.com>
 * 
 */

#pragma once

#include "cstone/primitives/primitives_acc.hpp"

template<class T>
class pinned_allocator;

namespace halo
{

//! @brief std::allocator on the CPU, pinned_allocator on the GPU
template<class Accelerator, class T>
using PinnedAlloc_t = std::conditional_t<cstone::HaveGpu<Accelerator>{}, pinned_allocator<T>, std::allocator<T>>;

//! @brief stub for use in CPU code
struct DeviceHaloDataFacade
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

class DeviceHaloData;

//! @brief Just a facade on the CPU, DeviceHaloData on the GPU
template<class Accelerator>
using DeviceHaloData_t = std::conditional_t<cstone::HaveGpu<Accelerator>{}, DeviceHaloData, DeviceHaloDataFacade>;

} // namespace cluster