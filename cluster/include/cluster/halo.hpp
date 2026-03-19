
#pragma once

#include "halo_gpu.h"

namespace cluster
{
template<class ParticleDataset, class ClusterDataset, class HaloDataset>
void computeHaloPropertiesLocal(
    size_t first,
    size_t last,
    ParticleDataset& d,
    ClusterDataset& c,
    HaloDataset& h)
{
    if constexpr (cstone::HaveGpu<typename ParticleDataset::AcceleratorType>{})
        { haloPropertiesGPU(first, last, d, c, h); }
    else { printf("Error: No CPU implementation of halo properties computation available\n"); }
}

template<class HaloDataset>
void communicateHaloProperties(
    size_t first,
    size_t last,
    HaloDataset& h)
{
    if constexpr (cstone::HaveGpu<typename HaloDataset::AcceleratorType>{})
        { communicateHaloPropertiesGPU(first, last, h); }
    else { printf("Error: No CPU implementation of halo properties communication available\n"); }
}
}