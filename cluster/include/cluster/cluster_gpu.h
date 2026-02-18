#pragma once
/*!
 * @brief  Public declarations for GPU FoF clustering (kernel + host launcher)
 * @author Vincente Della Balda <vinc.dellabalda@gmail.com>
 * 
 * This header declares CUDA kernels for FoF clustering and the host-side
 *
 * The definitions live in cluster_gpu.cu.
 */

#include "cstone/sfc/box.hpp"     // Box is plain C++
#include "cstone/tree/octree.hpp" // OctreeNsView is plain C++
#include "cstone/traversal/groups.hpp"  // GroupView is plain C++

#include "sph/particles_data.hpp"
#include "sph/types.hpp"
#include "cluster_data.hpp"
#include "halo_data.hpp"

namespace cluster
{
    /*
     * @brief Host-side entry point to compute cluster IDs on the GPU.
     */
    template<class ParticleDataset, class ClusterDataset>
    void computeLocalClusterIdGPU(
                          const cstone::GroupView& grp,
                          ParticleDataset& d,
                          ClusterDataset& c,
                          const cstone::Box<typename ParticleDataset::RealType>& box
                        );
    
    template<class ParticleDataset, class ClusterDataset, class DomainType>
    void computeGlobalClusterIdGPU(
                          ParticleDataset& d,
                          ClusterDataset& c,
                          DomainType& domain
                        );
                        
    template<class ClusterDataset, class HaloDataset, class DomainType>
    void computeCompactClusterIdGPU(
                          ClusterDataset& c,
                          HaloDataset& h,
                          DomainType& domain
                        );

    template<class ClusterDataset, class HaloDataset, class DomainType>
    void prepareParticleClusterMapGPU(
                          ClusterDataset& c,
                          HaloDataset& h,
                          DomainType& domain
                        );

    template<class ParticleDataset, class ClusterDataset>
    size_t computeDensityMaxGPU(
        const cstone::GroupView& grp,
        ParticleDataset& d,
        ClusterDataset& c,
        const cstone::Box<typename ParticleDataset::RealType>& box);

    template<class ParticleDataset, class ClusterDataset>
    void computeDensityGroupsGPU(
        const cstone::GroupView& grp,
        ParticleDataset& d,
        ClusterDataset& c,
        const cstone::Box<typename ParticleDataset::RealType>& box);

    template<class ParticleDataset, class ClusterDataset, class HaloDataset, class DomainType>
    void growSOHalosGPU(
        DomainType& domain,
        ParticleDataset& d,
        ClusterDataset& c,
        HaloDataset& h,
        const int myRank,
        const int numRanks
    );

    
}