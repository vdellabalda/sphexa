

#include "cstone/traversal/find_neighbors.cuh"
#include "density_saddle_kernel.hpp"
#include "union_find_gpu.cuh"

#include "sph/kernels.hpp"

#include "definitions.h"


namespace cluster
{
    using namespace unionfind;

    using cstone::LocalIndex;
    using cstone::TreeNodeIndex;
    using cstone::TravConfig;
    using cstone::GpuConfig;

template<class IdType, class Tm, class Tfactor, class Tc, class KeyType, class Th>
__global__ void densitySaddlesGPU(
        const LocalIndex* grpStart, const LocalIndex* grpEnd, LocalIndex numGroups,
        const cstone::OctreeNsView<Tc, KeyType> tree, const cstone::Box<Tc> box,
        const Tc* x, const Tc* y, const Tc* z, const Th* h,
        const Tm* rho, const IdType* fofId, const IdType* densityZone, unsigned* nc, unsigned ngmax,
        Tfactor mergeFactor, IdType* parents, size_t numParticlesHalos, LocalIndex* nidx, TreeNodeIndex* globalPool
    )
{
    unsigned laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);
    unsigned targetIdx = 0;
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

        if (cstone::ballotSync(fofId[i]==0)) continue; // skip if no particles in FOF group

        unsigned ncSph =
             traverseNeighbors(bodyBegin, bodyEnd, x, y, z, h, tree, box, neighborsWarp, ngmax, globalPool)[0];

        if (i >= bodyEnd || fofId[i] == 0) continue; // skip if thread has no particle or particles not in FOF group
            
        unsigned neighborCount = ncSph;
        densitySaddleLoop<TravConfig::targetSize>(
            i, neighborsWarp + laneIdx, neighborCount, 
            fofId, densityZone, rho, mergeFactor, parents, numParticlesHalos);
    }
}

//template<class IdType, class Tm, class Tf>
//__global__ void mergeDensityZonesGPU(
//        const LocalIndex* grpStart, const LocalIndex* grpEnd, LocalIndex numGroups, size_t numParticlesHalos,
//        IdType* densityZone, const IdType* candidateZone, const Tm* candidateDensity, const Tm* rho, Tf mergeFactor, IdType* parents
//    )
//{
//    unsigned laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);
//    unsigned warpIdxGrid = (blockDim.x * blockIdx.x + threadIdx.x) >> GpuConfig::warpSizeLog2;
//    if (warpIdxGrid >= numGroups) return;
//
//    LocalIndex bodyBegin = grpStart[warpIdxGrid];
//    LocalIndex bodyEnd   = grpEnd[warpIdxGrid];
//    LocalIndex i         = bodyBegin + laneIdx;
//
//    if (i >= bodyEnd) return;
//
//    auto zoneId = densityZone[i];
//    auto candidateZoneId = candidateZone[i];
//    Tm minDensityPeak = stl::min(rho[zoneId], rho[candidateZoneId]);
//    bool mergeZones = (candidateZoneId != zoneId) && (candidateDensity[i] > mergeFactor * minDensityPeak);
//    if (mergeZones) {
//        uniteGPU(parents, zoneId, candidateZoneId, numParticlesHalos);
//    }
//
//}

template<class IdType>
__global__ void updateIdGPU(IdType* oldClusterIdx, const IdType* newClusterIdx, size_t numParticlesHalos)
{
    unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numParticlesHalos) return;

    IdType oldClusterId = oldClusterIdx[idx];
    IdType newClusterId = newClusterIdx[oldClusterId];
    oldClusterIdx[idx] = newClusterId;

}

} // namespace cluster