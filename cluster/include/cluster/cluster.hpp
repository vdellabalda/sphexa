#pragma once

#include "cluster_mpi.hpp"
#include "cluster_gpu.h"

namespace cluster
{
template<class DomainType, class Box, class ParticleDataset, class ClusterDataset>
void computeLocalClusterId(
    const cstone::GroupView& grp,
    ParticleDataset& d,
    ClusterDataset& c,
    Box& box,
    DomainType& domain
)
{
    if constexpr (cstone::HaveGpu<typename ParticleDataset::AcceleratorType>{})
        { computeLocalClusterIdGPU(grp, d, c, box); }
    else 
        { 
          computeLocalClusterIdImpl(grp.firstBody, grp.lastBody, d, c, box, domain);
        }
}

template<class ParticleDataset, class ClusterDataset, class DomainType>
void computeGlobalClusterId(
    ParticleDataset& d,
    ClusterDataset& c,
    DomainType& domain)
{
    if constexpr (cstone::HaveGpu<typename ParticleDataset::AcceleratorType>{})
        { 
            computeGlobalClusterIdGPU(d, c, domain);
        }
    else 
        { 
            computeGlobalClusterIdImpl(c, domain);
        }
}

template<class ClusterDataset, class HaloDataset, class DomainType>
void computeCompactClusterId(
    ClusterDataset& c,
    HaloDataset& h,
    DomainType& domain)
{
    if constexpr (cstone::HaveGpu<typename ClusterDataset::AcceleratorType>{})
        { computeCompactClusterIdGPU(c, h, domain); }
    else 
        { computeCompactClusterIdImpl(c, domain.startIndex(), domain.endIndex()); }
}

template<class ClusterDataset, class HaloDataset, class DomainType>
void prepareParticleClusterMap(
    ClusterDataset& c,
    HaloDataset& h,
    DomainType& domain)
{
    if constexpr (cstone::HaveGpu<typename ClusterDataset::AcceleratorType>{})
        { prepareParticleClusterMapGPU(c, h, domain); }
    else 
        { printf("prepareParticleClusterMap not implemented for CPU\n"); }
}

} // namespace cluster