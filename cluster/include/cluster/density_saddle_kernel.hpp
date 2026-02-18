#pragma once

#include "cstone/cuda/annotation.hpp"
#include "cstone/primitives/stl.hpp"

#include "union_find_gpu.cuh"


namespace cluster
{

using namespace unionfind;

// Goal: Amongst all neighbors particles of a given particle,
// determine whether the neighbor belongs to another density zone
// if so, merge the density zones if condition holds
template<size_t stride = 1, class IdType, class Tm, class Tfactor>
HOST_DEVICE_FUN inline void densitySaddleLoop(
    IdType i, const IdType* neighbors, unsigned neighborsCount, const IdType* fofId, const IdType* densityZoneId, const Tm* rho,
    Tfactor mergeFactor, IdType* parents, size_t numParticlesHalos
)
{
    auto rhoi = rho[i];
    auto zonei = densityZoneId[i];
    auto fofi = fofId[i];

    for (unsigned pj = 0; pj < neighborsCount; ++pj)
    {
        IdType j = neighbors[stride * pj];
        if (fofId[j] != fofi) continue; // Only consider neighbors in the same FOF group
        IdType zonej = densityZoneId[j];
        if (zonej != zonei)
        {
            Tm candidateDensity = stl::min(rhoi, rho[j]);
            bool mergeZones = candidateDensity > mergeFactor * stl::min(rho[zonei], rho[zonej]);
            if (mergeZones) {
                uniteGPU(parents, zonei, zonej, numParticlesHalos);
            }
        }
    }
}
}