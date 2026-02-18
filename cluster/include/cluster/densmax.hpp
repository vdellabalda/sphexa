#pragma once

#include "cluster_gpu.h"

namespace cluster
{
template<class ParticleDataset, class ClusterDataset, class T>
void computeDensityGroups(
    const cstone::GroupView& grp,
    ParticleDataset& d, ClusterDataset& c,
    const cstone::Box<T>& box
)
{
    if constexpr (cstone::HaveGpu<typename ParticleDataset::AcceleratorType>{})
        { computeDensityGroupsGPU(grp, d, c, box); }
    else { printf("computeDensityGroups not implemented for CPU\n"); }
}
}