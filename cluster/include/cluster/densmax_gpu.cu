
#include "cstone/traversal/find_neighbors.cuh"
#include "cstone/traversal/groups_gpu.cuh"
#include "cstone/primitives/mpi_cuda.cuh"
#include "cstone/domain/domain.hpp"

#include "sph/kernels.hpp"

#include "cluster_gpu.h"
#include "cluster_gpu_thrust.cuh"
#include "densmax_kern.hpp"
#include "density_saddle_gpu.cuh"
#include "union_find_gpu.cuh"

namespace cluster
{
    using namespace unionfind;

    using cstone::LocalIndex;
    using cstone::TreeNodeIndex;
    using cstone::TravConfig;
    using cstone::GpuConfig;


template<class IdType, class KeyType, class Tc, class Th, class Tm>
__global__ void densestFOFNeighborGPU(
        const LocalIndex* grpStart, const LocalIndex* grpEnd, LocalIndex numGroups,
        const cstone::OctreeNsView<Tc, KeyType> tree, const cstone::Box<Tc> box,
        unsigned ng0, unsigned ngmax, const IdType* fofId, const Tc* x, const Tc* y, const Tc* z,
        Th* h, Tm* rho, IdType* parent,
        LocalIndex* nidx, TreeNodeIndex* globalPool
    )
{
    unsigned laneIdx     = threadIdx.x & (GpuConfig::warpSize - 1);
    unsigned targetIdx   = 0;
    unsigned warpIdxGrid = (blockDim.x * blockIdx.x + threadIdx.x) >> GpuConfig::warpSizeLog2;

    LocalIndex* neighborsWarp = nidx + ngmax * TravConfig::targetSize * warpIdxGrid;

    while (true)
    {
        // first thread in warp grabs next target
        if (laneIdx == 0) { targetIdx = atomicAdd(&cstone::targetCounterGlob, 1); }
        targetIdx = cstone::shflSync(targetIdx, 0);

        if (targetIdx >= numGroups) return;

        LocalIndex bodyBegin = grpStart[targetIdx];
        LocalIndex bodyEnd   = grpEnd[targetIdx];
        LocalIndex i         = bodyBegin + laneIdx;

        unsigned ncSph =
            1 + traverseNeighbors(bodyBegin, bodyEnd, x, y, z, h, tree, box, neighborsWarp, ngmax, globalPool)[0];

        constexpr int ncMaxIteration = 9;
            for (int ncIt = 0; ncIt <= ncMaxIteration; ++ncIt)
            {
                bool repeat = (ncSph < ng0 / 4 || (ncSph - 1) > ngmax) && i < bodyEnd;
                if (!cstone::ballotSync(repeat)) { break; }
                if (repeat) { h[i] = sph::updateH(ng0, ncSph, h[i]); }
                ncSph =
                    1 + traverseNeighbors(bodyBegin, bodyEnd, x, y, z, h, tree, box, neighborsWarp, ngmax, globalPool)[0];

                bool ncFail = (ncSph < ng0 / 4 || (ncSph - 1) > ngmax) && i < bodyEnd;
                if (ncIt == ncMaxIteration && ncFail) 
                { 
                    printf("Warning: particle %u has nc=%u after %d iterations.\n", i, ncSph-1, ncIt);
                    ncSph = 1; 
                }
            }

        if (i >= bodyEnd) continue;
        if (fofId[i] == 0) continue; // skip if not in a FOF group
            
        auto ncCapped = stl::min(ncSph - 1, ngmax);
        //nc[i] = ncCapped;
        parent[i] = densestFOFNeighborLoop<TravConfig::targetSize>(i, neighborsWarp + laneIdx, ncCapped, fofId, rho);
    }
}

template<class IdType, class KeyType, class Tc, class Th, class Tm>
__global__ void densestNeighborGPU(
        const LocalIndex* grpStart, const LocalIndex* grpEnd, LocalIndex numGroups,
        const cstone::OctreeNsView<Tc, KeyType> tree, const cstone::Box<Tc> box,
        unsigned ng0, unsigned ngmax, const Tc* x, const Tc* y, const Tc* z,
        Th* h, Tm* rho, IdType* parent,
        LocalIndex* nidx, TreeNodeIndex* globalPool
    )
{
    unsigned laneIdx     = threadIdx.x & (GpuConfig::warpSize - 1);
    unsigned targetIdx   = 0;
    unsigned warpIdxGrid = (blockDim.x * blockIdx.x + threadIdx.x) >> GpuConfig::warpSizeLog2;

    LocalIndex* neighborsWarp = nidx + ngmax * TravConfig::targetSize * warpIdxGrid;

    while (true)
    {
        // first thread in warp grabs next target
        if (laneIdx == 0) { targetIdx = atomicAdd(&cstone::targetCounterGlob, 1); }
        targetIdx = cstone::shflSync(targetIdx, 0);

        if (targetIdx >= numGroups) return;

        LocalIndex bodyBegin = grpStart[targetIdx];
        LocalIndex bodyEnd   = grpEnd[targetIdx];
        LocalIndex i         = bodyBegin + laneIdx;

        unsigned ncSph =
            1 + traverseNeighbors(bodyBegin, bodyEnd, x, y, z, h, tree, box, neighborsWarp, ngmax, globalPool)[0];

        constexpr int ncMaxIteration = 9;
            for (int ncIt = 0; ncIt <= ncMaxIteration; ++ncIt)
            {
                bool repeat = (ncSph < ng0 / 4 || (ncSph - 1) > ngmax) && i < bodyEnd;
                if (!cstone::ballotSync(repeat)) { break; }
                if (repeat) { h[i] = sph::updateH(ng0, ncSph, h[i]); }
                ncSph =
                    1 + traverseNeighbors(bodyBegin, bodyEnd, x, y, z, h, tree, box, neighborsWarp, ngmax, globalPool)[0];

                bool ncFail = (ncSph < ng0 / 4 || (ncSph - 1) > ngmax) && i < bodyEnd;
                if (ncIt == ncMaxIteration && ncFail) 
                { 
                    printf("Warning: particle %u has nc=%u after %d iterations.\n", i, ncSph-1, ncIt);
                    ncSph = 1; 
                }
            }

        if (i >= bodyEnd) continue;
            
        auto ncCapped = stl::min(ncSph - 1, ngmax);
        //nc[i] = ncCapped;
        parent[i] = densestNeighborLoop<TravConfig::targetSize>(i, neighborsWarp + laneIdx, ncCapped, rho);
    }
}


template<class ParticleDataset, class ClusterDataSet>
void computeLocalDensityGroupsGPU(
    const cstone::GroupView& grp,
    ParticleDataset& d, ClusterDataSet& c,
    const cstone::Box<typename ParticleDataset::RealType>& box)
{
    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    LocalIndex numParticles = grp.lastBody - grp.firstBody;

    // Tighter neighbor search radius for density maxima and saddles
    unsigned ng0 = 40;
    unsigned ngmax = 80;
    float ngfac = pow(float(ng0)/float(d.ngmax), 1.f/3.f);

    auto [traversalPool, nidxPool] = cstone::allocateNcStacks(d.traversalStack, ngmax);
    cstone::resetTraversalCounters<<<1, 1>>>();

    cstone::scaleGpu(rawPtr(d.h)+grp.firstBody, rawPtr(d.h)+grp.lastBody, rawPtr(c.hTight)+grp.firstBody, ngfac);
    cstone::sequenceGpu(rawPtr(c.localClusterIds), c.numParticlesHalos, unsigned(0));
    
    //densestFOFNeighborGPU<<<TravConfig::numBlocks(), TravConfig::numThreads>>>(
    //    grp.groupStart, grp.groupEnd, grp.numGroups, d.treeView, box, ng0, ngmax,
    //    rawPtr(c.halo_id), rawPtr(d.x), rawPtr(d.y), rawPtr(d.z), rawPtr(c.hTight), rawPtr(d.rho),
    //    rawPtr(c.localClusterIds), nidxPool, traversalPool);
    //checkGpuErrors(cudaGetLastError());

    densestNeighborGPU<<<TravConfig::numBlocks(), TravConfig::numThreads>>>(
        grp.groupStart, grp.groupEnd, grp.numGroups, d.treeView, box, ng0, ngmax,
        rawPtr(d.x), rawPtr(d.y), rawPtr(d.z), rawPtr(c.hTight), rawPtr(d.rho),
        rawPtr(c.localClusterIds), nidxPool, traversalPool);
    checkGpuErrors(cudaGetLastError());

    // Update root of each particle
    unsigned numThreads = 128;
    unsigned numBlocks = (c.numParticlesHalos + numThreads - 1) / numThreads;
    if (numBlocks < 1) numBlocks = 1;
    updateRootGPU<<<numBlocks, numThreads>>>(rawPtr(c.localClusterIds), c.numParticlesHalos);
    checkGpuErrors(cudaGetLastError());

    //cstone::sequenceGpu(rawPtr(c.idBuf), c.numParticlesHalos, ClusterIdType(0));
//
    //cstone::resetTraversalCounters<<<1, 1>>>();
    //numBlocks = (numParticles + numThreads - 1) / numThreads;
    //if (numBlocks < 1) numBlocks = 1;
    //densitySaddlesGPU<<<numBlocks, numThreads>>>(
    //    grp.groupStart, grp.groupEnd, grp.numGroups,
    //    d.treeView, box, rawPtr(d.x), rawPtr(d.y), rawPtr(d.z), rawPtr(c.hTight),
    //    rawPtr(d.rho), rawPtr(c.halo_id), rawPtr(c.localClusterIds), rawPtr(d.nc), ngmax,
    //    c.mergeFactor, rawPtr(c.idBuf), c.numParticlesHalos, nidxPool, traversalPool
    //);
    //checkGpuErrors(cudaGetLastError());
//
    //updateRootGPU<<<numBlocks, numThreads>>>(rawPtr(c.idBuf), c.numParticlesHalos);
    //checkGpuErrors(cudaGetLastError());
//
    //// Update c.localClusterIds to reflect merged zones
    //updateIdGPU<<<numBlocks, numThreads>>>(rawPtr(c.localClusterIds), rawPtr(c.idBuf), c.numParticlesHalos);
    //checkGpuErrors(cudaGetLastError());
    
    transformLocalToGlobalClusterKeys(rawPtr(c.localClusterIds), rawPtr(c.globalClusterKeys), c.numParticlesHalos, myRank);
    checkGpuErrors(cudaGetLastError());    
}
template void computeLocalDensityGroupsGPU(
    const cstone::GroupView& grp, sphexa::ParticlesData<cstone::GpuTag>& d,
    cluster::ClusterData<cstone::GpuTag>& c,
    const cstone::Box<sph::SphTypes::CoordinateType>& box);


template<class ParticleDataset, class ClusterDataSet>
void computeGlobalDensityGroupsGPU(
    const cstone::GroupView& grp,
    ParticleDataset& d, ClusterDataSet& c,
    const cstone::Box<typename ParticleDataset::RealType>& box)
{
    printf("computeGlobalDensityGroups not implemented for GPU\n");
}
template void computeGlobalDensityGroupsGPU(
    const cstone::GroupView& grp, sphexa::ParticlesData<cstone::GpuTag>& d,
    cluster::ClusterData<cstone::GpuTag>& c,
    const cstone::Box<sph::SphTypes::CoordinateType>& box);


template<class ParticleDataset, class ClusterDataSet>
void computeDensitySaddlesGPU(
    const cstone::GroupView& grp,
    ParticleDataset& d, ClusterDataSet& c,
    const cstone::Box<typename ParticleDataset::RealType>& box)
{
    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    size_t numParticles = grp.lastBody - grp.firstBody;
    unsigned ng0 = 40;
    unsigned ngmax = 80;
    auto [traversalPool, nidxPool] = cstone::allocateNcStacks(d.traversalStack, ngmax);

    cstone::sequenceGpu(rawPtr(c.idBuf), c.numParticlesHalos, ClusterIdType(0));

    cstone::resetTraversalCounters<<<1, 1>>>();
    unsigned numThreads = 128;
    unsigned numBlocks = (numParticles + numThreads - 1) / numThreads;
    if (numBlocks < 1) numBlocks = 1;
    densitySaddlesGPU<<<numBlocks, numThreads>>>(
        grp.groupStart, grp.groupEnd, grp.numGroups,
        d.treeView, box, rawPtr(d.x), rawPtr(d.y), rawPtr(d.z), rawPtr(c.hTight),
        rawPtr(d.rho), rawPtr(c.halo_id), rawPtr(c.localClusterIds), rawPtr(d.nc), ngmax,
        c.mergeFactor, rawPtr(c.idBuf), c.numParticlesHalos, nidxPool, traversalPool
    );
    checkGpuErrors(cudaGetLastError());

    updateRootGPU<<<numBlocks, numThreads>>>(rawPtr(c.idBuf), c.numParticlesHalos);
    checkGpuErrors(cudaGetLastError());

    // Update c.localClusterIds to reflect merged zones
    updateIdGPU<<<numBlocks, numThreads>>>(rawPtr(c.localClusterIds), rawPtr(c.idBuf), c.numParticlesHalos);
    checkGpuErrors(cudaGetLastError());
    
    transformLocalToGlobalClusterKeys(rawPtr(c.localClusterIds), rawPtr(c.globalClusterKeys), c.numParticlesHalos, myRank);
    checkGpuErrors(cudaGetLastError());
}
} // namespace cluster