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
    const int myRank,
    DomainType& domain
)
{
    if constexpr (cstone::HaveGpu<typename ParticleDataset::AcceleratorType>{})
        { computeLocalClusterIdGPU(grp, d, c, box, myRank); }
    else 
        { 
          computeLocalClusterIdImpl(grp.firstBody, grp.lastBody, d, c, box, myRank, domain);
        }
}

template<class ParticleDataset, class ClusterDataset, class DomainType>
void computeGlobalClusterId(
    ParticleDataset& d,
    ClusterDataset& c,
    DomainType& domain,
    const int myRank)
{
    if constexpr (cstone::HaveGpu<typename ParticleDataset::AcceleratorType>{})
        { 
            computeGlobalClusterIdGPU(d, c, domain, myRank);
        }
    else 
        { 
            computeGlobalClusterIdImpl(c, domain, myRank);
        }
}


template<class ClusterDataset, class DomainType>
void computeCompactClusterId(
    ClusterDataset& c,
    DomainType& domain,
    const int myRank,
    const int numRanks)
{
    if constexpr (cstone::HaveGpu<typename ClusterDataset::AcceleratorType>{})
        { computeCompactClusterIdGPU(c, domain, myRank, numRanks); }
    else 
        { computeCompactClusterIdImpl(c, domain.startIndex(), domain.endIndex(), myRank, numRanks); }
}
} // namespace cluster