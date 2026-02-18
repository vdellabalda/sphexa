#pragma once

#include "cstone/cuda/annotation.hpp"

namespace cluster
{

template<size_t stride = 1, class IdType, class Tm>
HOST_DEVICE_FUN inline unsigned densityMaxLoop(
    IdType i, const IdType* neighbors, unsigned neighborsCount, const Tm* rho)
{
    auto rhoi = rho[i];
    unsigned isMax = 1;
    Tm meanRho = 0;

    for (unsigned pj = 0; pj < neighborsCount; ++pj)
    {
        IdType j = neighbors[stride * pj];
        meanRho += rho[j];
        if ((rho[j] > rhoi) && (rho[j] > 2*meanRho))
        {
            isMax = 0;
            break;
        }
    }
    meanRho /= Tm(neighborsCount);
    if (rhoi < 2*meanRho)
        isMax = 0;

    return isMax;
}

template<size_t stride = 1, class IdType, class Tm>
HOST_DEVICE_FUN inline IdType densestFOFNeighborLoop(
    IdType i, const IdType* neighbors, unsigned neighborsCount, const IdType* FOFId, const Tm* rho)
{
    auto rhoi = rho[i];
    auto fofi = FOFId[i];
    IdType densestNeighbor = i;

    for (unsigned pj = 0; pj < neighborsCount; ++pj)
    {
        IdType j = neighbors[stride * pj];
        if (rho[j] > rhoi && FOFId[j] == fofi)
        {
            rhoi = rho[j];
            densestNeighbor = j;
        }
    }

    return densestNeighbor;
}
}