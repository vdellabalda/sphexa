/*! @file
 * @brief Friends of friends algorithm for halo finding
 *
 * This file implements the friends of friends (FoF) algorithm for halo finding in cosmological simulations.
 * The algorithm is designed to work with MPI and uses a tree-based approach to find halos in a distributed system.
 * It includes GPU kernels for efficient computation of particle-particle interactions and cluster identification.
 * 
 * @author Vincente Della Balda
 */

#include "cstone/fields/field_get.hpp"
#include "cstone/traversal/find_neighbors.cuh"
#include "cstone/traversal/groups_gpu.cuh"
#include "cstone/primitives/mpi_cuda.cuh"
#include "cstone/domain/domain.hpp"

#include "cluster_utils.hpp"
#include "hashmap_gpu.h"
#include "cluster_gpu.h"
#include "cluster_gpu_thrust.cuh"
#include "union_find_gpu.cuh"

#include "definitions.h"

namespace cstone

{
using namespace unionfind;


/*! @brief collect edges between target bodies and source bodies within cutoff distance
 *
 * @tparam       Tc            float or double
 * @param[in]    sourceBody    source body x,y,z
 * @param[in]    validLaneMask number of lanes that contain valid source bodies
 * @param[in]    pos_i         target body x,y,z,cutoff^2
 * @param[in]    box           global coordinate bounding box
 * @param[in]    targetBodyIdx index of target body of each lane
 * @param[in]    sourceBodyIdx index of source body of each lane
 * @param[inout] ec_i          warp edge counts to add to
 * @param[inout] eidx_i        target body indeces of edges
 * @param[inout] eidx_j        source body indeces of edges
 *
 * Number of computed particle-particle pairs per call is GpuConfig::warpSize^2 * TravConfig::nwt
 */
template<bool UsePbc, class Tc, class Index>
__device__ size_t edgeCollector(
    Vec3<Tc> sourceBody,
    int numLanesValid,
    const util::array<Vec4<Tc>, TravConfig::nwt>& pos_i,
    const Box<Tc>& box,
    const Index targetBodyIdx[TravConfig::nwt],
    LocalIndex sourceBodyIdx,
    size_t ec_i,
    LocalIndex* clusterId,
    LocalIndex* clusterChanged,
    LocalIndex numParticles) 
{   
    size_t collectedEdges = 0;

    for (int j = 0; j < numLanesValid; j++)
    {   
        Vec3<Tc> pos_j{shflSync(sourceBody[0], j), shflSync(sourceBody[1], j), shflSync(sourceBody[2], j)};
        LocalIndex idx_j = shflSync(sourceBodyIdx, j);

#pragma unroll
        for (int k = 0; k < TravConfig::nwt; k++)
        {   
            Tc d2 = distanceSq<UsePbc>(pos_j[0], pos_j[1], pos_j[2], pos_i[k][0], pos_i[k][1], pos_i[k][2], box);
            //if ((d2 < pos_i[k][3]) && ((idx_j > targetBodyIdx[k]) || (idx_j < startIndex)));
            if (d2 < pos_i[k][3])
            {
                uniteGPU(clusterId, targetBodyIdx[k], idx_j, numParticles);
                atomicOr(&clusterChanged[targetBodyIdx[k]], 1);
                atomicOr(&clusterChanged[idx_j], 1);
                collectedEdges++;
            }
        }
    }
    
    auto inclusiveShift = inclusiveScanInt(collectedEdges);
    ec_i += shflSync(inclusiveShift, GpuConfig::warpSize - 1);
    return ec_i;
}


/*! @brief traverse one warp with up to TravConfig::targetSize target bodies down the tree
 *
 * @param[inout]    clusterId      cluster Label array
 * @param[inout]    changedHaloId  flag clustered particles
 * @param[in]    pos_i          target x,y,z,percLength^2, TravConfig::nwt per lane
 * @param[in]    targetCenter   geometrical target center
 * @param[in]    targetSize     geometrical target bounding box size
 * @param[in]    targetBodyIdx  index of pos_i
 * @param[in]    x,y,z          source bodies as referenced by tree cells
 * @param[in]    tree           octree data view
 * @param[in]    initNodeIdx    traversal will be started with all children of the parent of @p initNodeIdx
 * @param[in]    depth          depth of the octree node to start traversal at
 * @param[in]    box            global coordinate bounding box
 * @param[-]     tempQueue      shared mem int pointer to GpuConfig::warpSize ints, uninitialized
 * @param[-]     cellQueue      pointer to global memory, size defined by TravConfig::memPerWarp, uninitialized
 *
 */
template<bool UsePbc, class Tc, class KeyType, class Index>
__device__ uint3 traverseWarpDSU(
                                 LocalIndex* clusterId,
                                 LocalIndex* changedHaloId,
                                 const util::array<Vec4<Tc>, TravConfig::nwt>& pos_i,
                                 const Vec3<Tc> targetCenter,
                                 const Vec3<Tc> targetSize,
                                 const Index targetBodyIdx[TravConfig::nwt],
                                 const Tc* __restrict__ x,
                                 const Tc* __restrict__ y,
                                 const Tc* __restrict__ z,
                                 const OctreeNsView<Tc, KeyType>& tree,
                                 int initNodeIdx,
                                 const Box<Tc>& box,
                                 volatile int* tempQueue,
                                 int* cellQueue,
                                 LocalIndex numParticles)
{
    const TreeNodeIndex* __restrict__ childOffsets   = tree.childOffsets;
    const TreeNodeIndex* __restrict__ internalToLeaf = tree.internalToLeaf;
    const LocalIndex* __restrict__ layout            = tree.layout;
    const Vec3<Tc>* __restrict__ centers             = tree.centers;
    const Vec3<Tc>* __restrict__ sizes               = tree.sizes;

    const int laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);    

    unsigned p2pCounter = 0, maxStack = 0, edgeCounter = 0;

    int bodyQueue; // warp queue for source body indices

    // populate initial cell queue
    if (laneIdx == 0) { cellQueue[0] = initNodeIdx; }

    // these variables are always identical on all warp lanes
    int numSources   = 1; // current stack size
    int newSources   = 0; // stack size for next level
    int oldSources   = 0; // cell indices done
    int sourceOffset = 0; // current level stack pointer, once this reaches numSources, the level is done
    int bdyFillLevel = 0; // fill level of the source body warp queue

    while (numSources > 0) // While there are source cells to traverse
    {
        int sourceIdx   = sourceOffset + laneIdx; // Source cell index of current lane
        int sourceQueue = 0;
        if (laneIdx < GpuConfig::warpSize / 8)
        {
            sourceQueue = cellQueue[ringAddr(oldSources + sourceIdx)]; // Global source cell index in queue
        }
        sourceQueue         = spreadSeg8(sourceQueue);
        sourceIdx           = shflSync(sourceIdx, laneIdx >> 3);
        const bool isSource = sourceIdx < numSources; // Source index is within bounds
        if (!isSource) { sourceQueue = 0; }

        const Vec3<Tc> curSrcCenter = centers[sourceQueue];      // Current source cell center
        const Vec3<Tc> curSrcSize   = sizes[sourceQueue];        // Current source cell center
        const int childBegin        = childOffsets[sourceQueue]; // First child cell
        const bool isNode           = childBegin;
        const bool isClose          = cellOverlap<UsePbc>(curSrcCenter, curSrcSize, targetCenter, targetSize, box);
        const bool isDirect         = isClose && !isNode && isSource;
        const int leafIdx           = (isDirect) ? internalToLeaf[sourceQueue] : 0; // the cstone leaf index

        // Split
        const bool isSplit     = isNode && isClose && isSource;                   // Source cell must be split
        const int numChildLane = exclusiveScanBool(isSplit);                      // Exclusive scan of numChild
        const int numChildWarp = reduceBool(isSplit);                             // Total numChild of current warp
        sourceOffset += imin(GpuConfig::warpSize / 8, numSources - sourceOffset); // advance current level stack pointer
        int childIdx = oldSources + numSources + newSources + numChildLane;       // Child index of current lane
        if (isSplit) { cellQueue[ringAddr(childIdx)] = childBegin; }              // Queue child cells for next level
        newSources += numChildWarp; // Increment source cell count for next loop

        // check for cellQueue overflow
        const unsigned stackUsed = newSources + numSources - sourceOffset; // current cellQueue size
        maxStack                 = max(stackUsed, maxStack);
        if (stackUsed > TravConfig::memPerWarp) { return {0xFFFFFFFF, maxStack}; } // Exit if cellQueue overflows

        // Direct
        const int firstBody     = layout[leafIdx];
        const int numBodies     = (layout[leafIdx + 1] - firstBody) & -int(isDirect); // Number of bodies in cell
        bool directTodo         = numBodies;
        const int numBodiesScan = inclusiveScanInt(numBodies);                      // Inclusive scan of numBodies
        int numBodiesLane       = numBodiesScan - numBodies;                        // Exclusive scan of numBodies
        int numBodiesWarp       = shflSync(numBodiesScan, GpuConfig::warpSize - 1); // Total numBodies of current warp
        int prevBodyIdx         = 0;
        //bool edgeMemFilled      = 0;
        while (numBodiesWarp > 0) // While there are bodies to process from current source cell set
        {
            tempQueue[laneIdx] = 1; // Default scan input is 1, such that consecutive lanes load consecutive bodies
            if (directTodo && (numBodiesLane < GpuConfig::warpSize))
            {
                directTodo               = false;          // Set cell as processed
                tempQueue[numBodiesLane] = -1 - firstBody; // Put first source cell body index into the queue
            }
            const int bodyIdx = inclusiveSegscanInt(tempQueue[laneIdx], prevBodyIdx);
            // broadcast last processed bodyIdx from the last lane to restart the scan in the next iteration
            prevBodyIdx = shflSync(bodyIdx, GpuConfig::warpSize - 1);
            
            if (numBodiesWarp >= GpuConfig::warpSize) // Process bodies from current set of source cells
            {
                // Load source body coordinates
                const Vec3<Tc> sourceBody = {x[bodyIdx], y[bodyIdx], z[bodyIdx]};
                edgeCounter = edgeCollector<UsePbc>(sourceBody, GpuConfig::warpSize, pos_i, box, targetBodyIdx, bodyIdx,
                    edgeCounter, clusterId, changedHaloId, numParticles);
                numBodiesWarp -= GpuConfig::warpSize;
                numBodiesLane -= GpuConfig::warpSize;
                p2pCounter += GpuConfig::warpSize;
                
            }
            else // Fewer than warpSize bodies remaining from current source cell set
            {
                // push the remaining bodies into bodyQueue
                int topUp = shflUpSync(bodyIdx, bdyFillLevel);
                bodyQueue = (laneIdx < bdyFillLevel) ? bodyQueue : topUp;

                bdyFillLevel += numBodiesWarp;
                if (bdyFillLevel >= GpuConfig::warpSize) // If this causes bodyQueue to spill
                {
                    // Load source body coordinates
                    const Vec3<Tc> sourceBody = {x[bodyQueue], y[bodyQueue], z[bodyQueue]};
                    edgeCounter = edgeCollector<UsePbc>(sourceBody, GpuConfig::warpSize, pos_i, box, targetBodyIdx, bodyQueue,
                        edgeCounter, clusterId, changedHaloId, numParticles);
                    bdyFillLevel -= GpuConfig::warpSize;
                    // bodyQueue is now empty; put body indices that spilled into the queue
                    bodyQueue = shflDownSync(bodyIdx, numBodiesWarp - bdyFillLevel);
                    p2pCounter += GpuConfig::warpSize;
                }
                numBodiesWarp = 0; // No more bodies to process from current source cells
            }
        }

        //  If the current level is done
        if (sourceOffset >= numSources)
        {
            oldSources += numSources;      // Update finished source size
            numSources   = newSources;     // Update current source size
            sourceOffset = newSources = 0; // Initialize next source size and offset
        }
    }

    if (bdyFillLevel > 0) // If there are leftover direct bodies
    {
        // Load position of source bodies, with padding for invalid lanes
        const bool laneHasBody = laneIdx < bdyFillLevel;
        const Vec3<Tc> sourceBody =
            laneHasBody ? Vec3<Tc>{x[bodyQueue], y[bodyQueue], z[bodyQueue]} : Vec3<Tc>{Tc(0), Tc(0), Tc(0)};
            edgeCounter = edgeCollector<UsePbc>(sourceBody, bdyFillLevel, pos_i, box, targetBodyIdx, bodyQueue,
                edgeCounter, clusterId, changedHaloId, numParticles);
            p2pCounter += bdyFillLevel;        
    }

    return {p2pCounter, maxStack, edgeCounter};
}


//! @brief edge search traversal statistics
struct EcStats
{
    using type = unsigned long long;
    enum IndexNames
    {
        sumP2P,
        sumEdges,
        maxP2P,
        maxStack,
        numStats
    };
};
static __device__ EcStats::type ecStats[EcStats::numStats];

static __device__ unsigned targetCounterGlobFOF;

static __global__ void resetEdgeTraversalCounters()
{
    for (int i = 0; i < EcStats::numStats; ++i)
    {
        ecStats[i] = 0;
    }

    targetCounterGlobFOF = 0;
}

template<class Tc, class Index>
__device__ __forceinline__ util::array<Vec4<Tc>, TravConfig::nwt> loadTargetFOF(
    Index bodyBegin,
    Index bodyEnd,
    Index bodyIdx[TravConfig::nwt],
    unsigned lane,
    const Tc* __restrict__ x,
    const Tc* __restrict__ y,
    const Tc* __restrict__ z,
    const Tc radius)
{
    util::array<Vec4<Tc>, TravConfig::nwt> pos_i;
#pragma unroll
    for (int i = 0; i < TravConfig::nwt; i++)
    {
        bodyIdx[i] = imin(bodyBegin + i * GpuConfig::warpSize + lane, bodyEnd - 1);
        pos_i[i]      = {x[bodyIdx[i]], y[bodyIdx[i]], z[bodyIdx[i]], radius};
    }
    return pos_i;
}

/*! @brief Traverse octree to find neighbors, resp. edges, within cutoff distance for FoF clustering
 *
 * @param[in]  bodyBegin   index of first particle in (x,y,z) to look for neighbors
 * @param[in]  bodyEnd     last (excluding) index of particle to look for neighbors
 * @param[in]  x           particle x coordinates
 * @param[in]  y           particle y coordinates
 * @param[in]  z           particle z coordinates
 * @param[in]  tree        octree connectivity and cell data
 * @param[in]  box         global coordinate bounding box
 * @param[in]  radius      cutoff radius for edge search
 * @param[inout]  clusterId   cluster Label array
 * @param[inout]  changedHaloId   flag clustered particles
 * @param[-]   globalPool  global memory for cell traversal stack
 *
 * Note: Number of handled particles (bodyEnd - bodyBegin) should be GpuConfig::warpSize * TravConfig::nwt or smaller
 */
template<class Tc, class KeyType>
__device__ void traverseNeighborsDSU(LocalIndex bodyBegin,
                                     LocalIndex bodyEnd,
                                     const Tc* __restrict__ x,
                                     const Tc* __restrict__ y,
                                     const Tc* __restrict__ z,
                                     const OctreeNsView<Tc, KeyType>& tree,
                                     const Box<Tc>& box,
                                     const Tc radius,
                                     LocalIndex* clusterId,
                                     LocalIndex* changedClusterId,
                                     TreeNodeIndex* globalPool,
                                     LocalIndex numParticles)
{
    const unsigned laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);
    const unsigned warpIdx = threadIdx.x >> GpuConfig::warpSizeLog2;

    constexpr unsigned numWarpsPerBlock = TravConfig::numThreads / GpuConfig::warpSize;

    __shared__ int sharedPool[TravConfig::numThreads];

    // warp-common shared mem, 1 int per thread
    int* tempQueue = sharedPool + GpuConfig::warpSize * warpIdx;
    // warp-common global mem storage
    int* cellQueue = globalPool + TravConfig::memPerWarp * ((blockIdx.x * numWarpsPerBlock) + warpIdx);

    LocalIndex bodyIdx[TravConfig::nwt];
    util::array<Vec4<Tc>, TravConfig::nwt> pos_i = loadTargetFOF(bodyBegin, bodyEnd, bodyIdx, laneIdx, x, y, z, radius);

    auto [targetCenter, targetSize]              = warpBbox(pos_i);
    auto r2 = radius * radius;

#pragma unroll
    for (int k = 0; k < TravConfig::nwt; ++k)
    {
        pos_i[k][3] = r2;
    }

    auto pbc    = BoundaryType::periodic;
    bool anyPbc = box.boundaryX() == pbc || box.boundaryY() == pbc || box.boundaryZ() == pbc;
    bool usePbc = anyPbc && !insideBox(targetCenter, targetSize, box);

    // start traversal with node 1 (first child of the root), implies siblings as well
    // if traversal should be started at node x, then initNode should be set to the first child of x
    int initNode = 1;

    uint3 warpStats;
    if (usePbc)
    {
        warpStats = traverseWarpDSU<true>(
            clusterId, changedClusterId,
            pos_i, targetCenter, targetSize, bodyIdx,
            x, y, z,
            tree, initNode, box, tempQueue, cellQueue, numParticles);
    }
    else
    {
        warpStats = traverseWarpDSU<false>(
            clusterId, changedClusterId,
            pos_i, targetCenter, targetSize, bodyIdx,
            x, y, z,
            tree, initNode, box, tempQueue, cellQueue, numParticles);
    }

    unsigned numP2P   = warpStats.x;
    unsigned maxStack = warpStats.y;
    unsigned edgeCount = warpStats.z;
    assert(numP2P != 0xFFFFFFFF);

    if (laneIdx == 0)
    {
        unsigned targetGroupSize = bodyEnd - bodyBegin;
        atomicAdd(&ecStats[EcStats::sumP2P], EcStats::type(numP2P) * targetGroupSize);
        atomicMax(&ecStats[EcStats::maxP2P], EcStats::type(numP2P));
        atomicMax(&ecStats[EcStats::maxStack], EcStats::type(maxStack));
        atomicAdd(&ecStats[EcStats::sumEdges], EcStats::type(edgeCount));
    }
}
}

namespace cluster
{

using namespace unionfind;

using cstone::GpuConfig;
using cstone::LocalIndex;
using cstone::TravConfig;
using cstone::TreeNodeIndex;

using util::FieldList;

template<class Tc, class KeyType>
__global__ void clusterIdGPU(
    const cstone::Box<Tc> box, const Tc radius,
    const LocalIndex* grpStart, const LocalIndex* grpEnd, LocalIndex numGroups,
    const cstone::OctreeNsView<Tc, KeyType> tree,
    const Tc* x, const Tc* y, const Tc* z,
    LocalIndex* clusterId,
    LocalIndex* changedClusterId,
    TreeNodeIndex* globalPool,
    LocalIndex numParticles)
{

    const unsigned laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);
    int targetIdx             = 0;

    while (true)
    {
        // first thread in warp grabs next target
        if (laneIdx == 0) { targetIdx = atomicAdd(&cstone::targetCounterGlobFOF, 1); }
        targetIdx = cstone::shflSync(targetIdx, 0);

        if (targetIdx >= numGroups) { return; }

        LocalIndex bodyBegin = grpStart[targetIdx];
        LocalIndex bodyEnd   = grpEnd[targetIdx];

        cstone::traverseNeighborsDSU(
            bodyBegin, bodyEnd,
            x, y, z,
            tree, box,
            radius, clusterId, changedClusterId,
            globalPool, numParticles);
    }
}


template<class ParticleDataset, class ClusterDataset>
__host__ void computeLocalClusterIdGPU(
    const cstone::GroupView& grp,
    ParticleDataset& d, ClusterDataset& c,
    const cstone::Box<typename ParticleDataset::RealType>& box)
{
    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    auto [traversalPool, emptyPool] = cstone::allocateNcStacks(d.traversalStack, 0);
    cstone::resetEdgeTraversalCounters<<<1, 1>>>();
    checkGpuErrors(cudaGetLastError());
    
    size_t numParticlesHalos = c.getNumParticlesHalos();
    size_t numParticles = grp.lastBody - grp.firstBody;
    auto percLength = c.getPercLength();
    c.numClusters.resize(1);
    cstone::fillGpu(rawPtr(c.flagged), rawPtr(c.flagged)+numParticlesHalos, unsigned(0));
    cstone::sequenceGpu(rawPtr(c.work_id), numParticlesHalos, ClusterIdType(0));

    clusterIdGPU<<<cstone::TravConfig::numBlocks(), TravConfig::numThreads>>>(
        box, percLength,
        grp.groupStart, grp.groupEnd, grp.numGroups,
        d.treeView,
        rawPtr(d.x), rawPtr(d.y), rawPtr(d.z),
        rawPtr(c.work_id), rawPtr(c.flagged),
        traversalPool, numParticlesHalos);
    checkGpuErrors(cudaGetLastError());

    cstone::EcStats::type stats[cstone::EcStats::numStats];
    checkGpuErrors(cudaMemcpyFromSymbol(stats, GPU_SYMBOL(cstone::ecStats),
                cstone::EcStats::numStats * sizeof(cstone::EcStats::type)));

    cstone::EcStats::type maxP2P   = stats[cstone::EcStats::maxP2P];
    cstone::EcStats::type maxStack = stats[cstone::EcStats::maxStack];
    cstone::EcStats::type sumEdges = stats[cstone::EcStats::sumEdges];

    c.edgesFoundEc = sumEdges;
    c.stackUsedEc = maxStack;

    if (maxP2P == 0xFFFFFFFF) { throw std::runtime_error("GPU traversal stack exhausted in neighbor search\n"); }
    
    unsigned numThreads = 256;
    unsigned numBlocks  = (numParticlesHalos + numThreads - 1) / numThreads;
    if (numBlocks < 1) numBlocks = 1;
    updateRootGPU<<<numBlocks, numThreads>>>(rawPtr(c.work_id), numParticlesHalos);
    checkGpuErrors(cudaGetLastError());

    transformLocalToGlobalClusterKeys(rawPtr(c.work_id),rawPtr(c.localClusterKeys),
        numParticlesHalos, myRank);
    checkGpuErrors(cudaGetLastError());
    memcpyD2D(rawPtr(c.localClusterKeys)+grp.firstBody, grp.lastBody-grp.firstBody,
        rawPtr(c.globalClusterKeys)+grp.firstBody);

    c.thresholdMask.reserve(numParticles);
    c.thresholdMask.resize(numParticles);
    cstone::fillGpu(rawPtr(c.thresholdMask), rawPtr(c.thresholdMask)+numParticles, unsigned(0));
    return;
}
template void computeLocalClusterIdGPU(
    const cstone::GroupView& grp,
    sphexa::ParticlesData<cstone::GpuTag>& d,
    cluster::ClusterData<cstone::GpuTag>& c,
    const cstone::Box<sph::SphTypes::CoordinateType>& box
);


template<class ParticleDataset, class ClusterDataset, class DomainType>
__host__ void computeGlobalClusterIdGPU(
    ParticleDataset& d, ClusterDataset& c, DomainType& domain
)
{   
    unsigned numThreads = 256;

    LocalIndex nLocal = domain.nParticles();
    LocalIndex nHalo = domain.nParticlesWithHalos() - nLocal;
    LocalIndex nHaloStart = domain.startIndex();
    LocalIndex nHaloEnd = nHalo - nHaloStart;

    size_t numClusteredHaloStart = cstone::reduceGpu(rawPtr(c.flagged), nHaloStart, size_t(0));
    size_t numClusteredHaloEnd = cstone::reduceGpu(rawPtr(c.flagged)+nHaloStart+nLocal,
                                nHaloEnd, size_t(0));        
    
    // edges for union-find
    c.edgeSrc.resize(numClusteredHaloStart + numClusteredHaloEnd);
    c.edgeDst.resize(numClusteredHaloStart + numClusteredHaloEnd);
    c.edges.resize(numClusteredHaloStart + numClusteredHaloEnd);

    if (numClusteredHaloStart)
    {
        flagSelectGpu(
            rawPtr(c.globalClusterKeys),
            rawPtr(c.flagged),
            rawPtr(c.edgeSrc),
            nHaloStart,
            rawPtr(c.idBuf), domain.nParticlesWithHalos()
        );
        flagSelectGpu(
            rawPtr(c.localClusterKeys),
            rawPtr(c.flagged),
            rawPtr(c.edgeDst),
            nHaloStart,
            rawPtr(c.idBuf), domain.nParticlesWithHalos()
        );

    }
    if (numClusteredHaloEnd)
    {
        flagSelectGpu(
            rawPtr(c.globalClusterKeys)+domain.endIndex(),
            rawPtr(c.flagged)+domain.endIndex(),
            rawPtr(c.edgeSrc)+numClusteredHaloStart,
            nHaloEnd,
            rawPtr(c.idBuf), domain.nParticlesWithHalos()
        );
        flagSelectGpu(
            rawPtr(c.localClusterKeys)+domain.endIndex(),
            rawPtr(c.flagged)+domain.endIndex(),
            rawPtr(c.edgeDst)+numClusteredHaloStart,
            nHaloEnd,
            rawPtr(c.idBuf), domain.nParticlesWithHalos()
        );
    }

    unsigned numBlocks  = (numClusteredHaloStart + numClusteredHaloEnd + numThreads - 1) / numThreads;
    if (numBlocks < 1) numBlocks = 1;
    constructEdgesGpu<<<numBlocks, numThreads>>>(rawPtr(c.edgeSrc), rawPtr(c.edgeDst),
                    numClusteredHaloStart+numClusteredHaloEnd, rawPtr(c.edges));
    checkGpuErrors(cudaGetLastError());  

    cstone::sortGpu(rawPtr(c.edges), rawPtr(c.edges)+numClusteredHaloStart+numClusteredHaloEnd);    
    EdgeType* newEdgesEnd = cstone::uniqueGpu(rawPtr(c.edges),
                    rawPtr(c.edges)+numClusteredHaloStart+numClusteredHaloEnd);
    size_t numUniqueEdges = newEdgesEnd-rawPtr(c.edges);
    size_t doubledNumUniqueEdges = 2*numUniqueEdges;
    cstone::DeviceVector<ClusterKeyType> flattenedEdgeKeys(doubledNumUniqueEdges);
    numBlocks  = (numUniqueEdges + numThreads - 1) / numThreads;
    if (numBlocks < 1) numBlocks = 1;
    flattenEdgesGpu<<<numBlocks, numThreads>>>(rawPtr(c.edges), numUniqueEdges, rawPtr(flattenedEdgeKeys));
    checkGpuErrors(cudaGetLastError());

    // Allgather of edges
    int numRanks;
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    std::vector<int> recvCounts(numRanks), displs(numRanks);
    MPI_Allgather(&doubledNumUniqueEdges, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    displs[0] = 0;
    for (int i = 1; i < numRanks; ++i)
        displs[i] = displs[i - 1] + recvCounts[i - 1];
    int totalCount = displs.back() + recvCounts.back();
    c.uniqueKeys.resize(totalCount);
    totalCount /= 2;
    
    mpiAllgathervGpuDirect<true>(rawPtr(flattenedEdgeKeys), doubledNumUniqueEdges, rawPtr(c.uniqueKeys), recvCounts.data(), displs.data(), MPI_COMM_WORLD);

    numBlocks = (totalCount + numThreads - 1) / numThreads;
    if (numBlocks < 1) numBlocks = 1;
    splitEdgesGpu<<<numBlocks, numThreads>>>(rawPtr(c.uniqueKeys), totalCount, rawPtr(c.keyBuf), rawPtr(c.keyBuf)+totalCount);

    cstone::sortGpu(rawPtr(c.uniqueKeys), rawPtr(c.uniqueKeys)+2*totalCount);
    checkGpuErrors(cudaGetLastError());
    ClusterKeyType* newKeysEnd = cstone::uniqueGpu(rawPtr(c.uniqueKeys), rawPtr(c.uniqueKeys)+2*totalCount);
    checkGpuErrors(cudaGetLastError());
    size_t numUniqueEdgeKeys = newKeysEnd - rawPtr(c.uniqueKeys);
    
    c.uniqueIds.resize(numUniqueEdgeKeys);
    cstone::sequenceGpu(rawPtr(c.uniqueIds), numUniqueEdgeKeys, ClusterIdType(0));
    checkGpuErrors(cudaGetLastError());
    
    numBlocks  = (2*totalCount + numThreads - 1) / numThreads;
    if (numBlocks < 1) numBlocks = 1;
    binarySearch<<<numBlocks, numThreads>>>(
        rawPtr(c.uniqueKeys), rawPtr(c.uniqueIds), numUniqueEdgeKeys,
        rawPtr(c.keyBuf), rawPtr(c.idBuf), 2*totalCount);
    checkGpuErrors(cudaGetLastError());

    // perform union-find on edges
    c.clusterParents.resize(numUniqueEdgeKeys);
    cstone::sequenceGpu(rawPtr(c.clusterParents), numUniqueEdgeKeys, ClusterIdType(0));
    checkGpuErrors(cudaGetLastError());

    numBlocks  = (totalCount + numThreads - 1) / numThreads;
    if (numBlocks < 1) numBlocks = 1;                    
    unionFindGpu<<<numBlocks, numThreads>>>(rawPtr(c.clusterParents), rawPtr(c.idBuf), rawPtr(c.idBuf)+totalCount, totalCount, numUniqueEdgeKeys);
    checkGpuErrors(cudaGetLastError());    
    updateRootGPU<<<numBlocks, numThreads>>>(rawPtr(c.clusterParents), numUniqueEdgeKeys);
    checkGpuErrors(cudaGetLastError());
 
    numBlocks = (domain.nParticles() + numThreads - 1) / numThreads;
    if (numBlocks < 1) numBlocks = 1;
    binarySearchFlagged<<<numBlocks, numThreads>>>(
        rawPtr(c.uniqueKeys), rawPtr(c.uniqueIds), numUniqueEdgeKeys,
        rawPtr(c.localClusterKeys)+domain.startIndex(), rawPtr(c.work_id)+domain.startIndex(), rawPtr(c.idBuf), domain.nParticles());
    checkGpuErrors(cudaGetLastError());

    clusterKeyUpdate<<<numBlocks, numThreads>>>(
        rawPtr(c.localClusterKeys)+domain.startIndex(),
        rawPtr(c.idBuf), rawPtr(c.work_id)+domain.startIndex(), rawPtr(c.clusterParents), rawPtr(c.uniqueKeys), domain.nParticles());    
    checkGpuErrors(cudaGetLastError());

    // Roots of the union-find are all (global) non-pure clusters
    c.thresholdMask.resize(numUniqueEdgeKeys);
    c.nonLocalKeys.resize(numUniqueEdgeKeys);
    cstone::fillGpu(rawPtr(c.thresholdMask), rawPtr(c.thresholdMask)+numUniqueEdgeKeys, unsigned(0));
    numBlocks = (numUniqueEdgeKeys + numThreads - 1) / numThreads;
    if (numBlocks < 1) numBlocks = 1;
    flagRoots<<<numBlocks, numThreads>>>(rawPtr(c.clusterParents), rawPtr(c.thresholdMask), numUniqueEdgeKeys);
    flagSelectGpu(rawPtr(c.uniqueKeys), rawPtr(c.thresholdMask), rawPtr(c.nonLocalKeys), numUniqueEdgeKeys, rawPtr(c.idBuf), domain.nParticlesWithHalos());
    c.numClustersNonLocal = cstone::reduceGpu(rawPtr(c.thresholdMask), numUniqueEdgeKeys, size_t(0));

    return;
}
template void computeGlobalClusterIdGPU(
    sphexa::ParticlesData<cstone::GpuTag>& d,
    cluster::ClusterData<cstone::GpuTag>& c,
    cstone::Domain<sph::SphTypes::KeyType, sph::SphTypes::CoordinateType, cstone::GpuTag>& domain);


template<class ClusterDataset, class HaloDataset, class DomainType>
__host__ void computeCompactClusterIdGPU(
    ClusterDataset& c,
    HaloDataset& h,
    DomainType& domain)
{   
    int numRanks, myRank;
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    auto startIndex = domain.startIndex();
    auto endIndex = domain.endIndex();

    // Compact keys of clustered particles
    flagSelectGpu(
            rawPtr(c.localClusterKeys)+domain.startIndex(),
            rawPtr(c.flagged)+domain.startIndex(),
            rawPtr(c.globalClusterKeys),
            domain.nParticles(),
            rawPtr(c.keyBuf), domain.nParticlesWithHalos()
    );

    size_t numClustered = cstone::reduceGpu(rawPtr(c.flagged)+domain.startIndex(), domain.nParticles(), size_t(0));
    cstone::sortGpu(rawPtr(c.globalClusterKeys), rawPtr(c.globalClusterKeys)+numClustered, rawPtr(c.keyBuf));
    checkGpuErrors(cudaGetLastError());
    auto numUniqueKeys = cstone::uniqueCountGpu(rawPtr(c.globalClusterKeys), rawPtr(c.globalClusterKeys)+numClustered);
    checkGpuErrors(cudaGetLastError());
    c.uniqueKeys.resize(numUniqueKeys); 
    c.localKeyCounts.resize(numUniqueKeys);
    cstone::runLengthEncodeGpu(
        numClustered, rawPtr(c.globalClusterKeys),
        rawPtr(c.uniqueKeys), rawPtr(c.localKeyCounts),
        rawPtr(c.numClusters),
        rawPtr(c.keyBuf), domain.nParticlesWithHalos());
    checkGpuErrors(cudaGetLastError());

    // Find intersection between local clusters and all (global) non-pure clusters
    // To find local non-pure clusters
    size_t numNonLocalKeys = c.numClustersNonLocal;
    std::pair<ClusterKeyType*, ClusterIdType*> newNonLocalIterators = setIntersectionByKeyGpu(
        rawPtr(c.uniqueKeys), numUniqueKeys, rawPtr(c.nonLocalKeys), numNonLocalKeys,
        rawPtr(c.localKeyCounts), rawPtr(c.keyBuf), rawPtr(c.idBuf));
    size_t nonLocalCount = newNonLocalIterators.first - rawPtr(c.keyBuf);

    // Find set difference between local pure and non-pure clusters
    // To find local pure clusters
    std::pair<ClusterKeyType*, ClusterIdType*> newLocalIterators = setDifferenceByKeyGpu(
        rawPtr(c.uniqueKeys), numUniqueKeys, rawPtr(c.keyBuf), nonLocalCount, rawPtr(c.localKeyCounts),
        rawPtr(c.idBuf), rawPtr(c.keyBuf)+nonLocalCount, rawPtr(c.idBuf)+nonLocalCount);
    size_t localCount = newLocalIterators.first - (rawPtr(c.keyBuf)+nonLocalCount);
    
    // Mark local clusters above threshold for sending
    c.thresholdMask.resize(nonLocalCount+localCount);
    cstone::fillGpu(rawPtr(c.thresholdMask), rawPtr(c.thresholdMask)+nonLocalCount, unsigned(1));
    thresholdMaskGpu(rawPtr(c.idBuf)+nonLocalCount, rawPtr(c.idBuf)+nonLocalCount+localCount,
        rawPtr(c.thresholdMask)+nonLocalCount, c.getClusterThreshold());

    flagSelectGpu(rawPtr(c.keyBuf), rawPtr(c.thresholdMask), rawPtr(c.uniqueKeys),
        nonLocalCount+localCount, rawPtr(c.globalClusterKeys), domain.nParticlesWithHalos());
    flagSelectGpu(rawPtr(c.idBuf), rawPtr(c.thresholdMask), rawPtr(c.localKeyCounts),
        nonLocalCount+localCount, rawPtr(c.globalClusterKeys), domain.nParticlesWithHalos());
    
    // Allgather Keys + Key Counts
    size_t sendCount = cstone::reduceGpu(rawPtr(c.thresholdMask), nonLocalCount+localCount, size_t(0));
    std::vector<int> recvCounts(numRanks), displs(numRanks);
    MPI_Allgather(&sendCount, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    displs[0] = 0;
    for (int i = 1; i < numRanks; ++i)
        displs[i] = displs[i - 1] + recvCounts[i - 1];
    int totalCount = displs.back() + recvCounts.back();

    mpiAllgathervGpuDirect<true>(rawPtr(c.uniqueKeys), sendCount, rawPtr(c.globalClusterKeys), recvCounts.data(), displs.data(), MPI_COMM_WORLD);
    mpiAllgathervGpuDirect<true>(rawPtr(c.localKeyCounts), sendCount, rawPtr(c.idBuf), recvCounts.data(), displs.data(), MPI_COMM_WORLD);

    cstone::sortByKeyGpu(rawPtr(c.globalClusterKeys), rawPtr(c.globalClusterKeys)+totalCount, rawPtr(c.idBuf),
        rawPtr(c.globalClusterKeys)+totalCount, rawPtr(c.idBuf)+totalCount, rawPtr(c.keyBuf), domain.nParticlesWithHalos());
    numUniqueKeys = cstone::uniqueCountGpu(rawPtr(c.globalClusterKeys), rawPtr(c.globalClusterKeys)+totalCount);    
    c.uniqueKeys.resize(numUniqueKeys);
    c.localKeyCounts.resize(numUniqueKeys);
    c.thresholdMask.resize(numUniqueKeys);
    c.idMap.resize(numUniqueKeys);

    std::pair<ClusterKeyType*, ClusterIdType*> new_iterators = cstone::reduceByKeyGpu(
        rawPtr(c.globalClusterKeys), rawPtr(c.globalClusterKeys)+totalCount, rawPtr(c.idBuf), rawPtr(c.uniqueKeys), rawPtr(c.localKeyCounts));
    
    cstone::sortByKeyDescendGpu(rawPtr(c.localKeyCounts), rawPtr(c.localKeyCounts)+numUniqueKeys, rawPtr(c.uniqueKeys));

    // Index where cluster size drops below threshold
    auto numClusters = cstone::upperBoundReverseGpu(rawPtr(c.localKeyCounts), rawPtr(c.localKeyCounts)+numUniqueKeys, c.getClusterThreshold());
    cstone::fillGpu(rawPtr(c.idMap), rawPtr(c.idMap)+numUniqueKeys, ClusterIdType(0));
    cstone::sequenceGpu(rawPtr(c.idMap), numClusters, ClusterIdType(1));

    cstone::sortByKeyGpu(rawPtr(c.uniqueKeys), rawPtr(c.uniqueKeys)+numUniqueKeys, rawPtr(c.idMap), rawPtr(c.keyBuf), rawPtr(c.idBuf),
        rawPtr(c.globalClusterKeys), domain.nParticlesWithHalos());

    // Use direct binary search to remap local cluster keys to global compact cluster Ids
    unsigned numThreads = 256;
    unsigned numBlocks = (domain.nParticles() + numThreads - 1) / numThreads;
    if (numBlocks < 1) numBlocks = 1;
    directClusterRemapping<<<numBlocks, numThreads>>>(
        rawPtr(c.localClusterKeys)+domain.startIndex(), 
        rawPtr(c.work_id)+domain.startIndex(),
        rawPtr(c.flagged)+domain.startIndex(),
        rawPtr(c.uniqueKeys), // already sorted
        rawPtr(c.idMap),      // corresponding compact IDs
        domain.nParticles(),
        numUniqueKeys
    );
    checkGpuErrors(cudaGetLastError());
    if (c.firstIter)
    {
        memcpyD2D(rawPtr(c.work_id)+domain.startIndex(), domain.nParticles(),
            rawPtr(c.halo_id)+domain.startIndex());
        c.firstIter = false;
    }
    else
    {
        memcpyD2D(rawPtr(c.work_id)+domain.startIndex(), domain.nParticles(),
            rawPtr(c.sub_id)+domain.startIndex());
        c.firstIter = true;
    }

    c.numClustersGlobal = numClusters;
}
template void computeCompactClusterIdGPU(
    cluster::ClusterData<cstone::GpuTag>& c,
    cluster::HaloData<cstone::GpuTag>& h,
    cstone::Domain<sph::SphTypes::KeyType, sph::SphTypes::CoordinateType, cstone::GpuTag>& domain
);

template<class ClusterDataset, class HaloDataset, class DomainType>
__host__ void prepareParticleClusterMapGPU(
    ClusterDataset& c,
    HaloDataset& h,
    DomainType& domain
)
{
    auto numClusters = c.numClustersGlobal;
    h.numClustersGlobal = numClusters;
    cstone::DeviceVector<LocalIndex> tempBuf(domain.nParticles());
    
    // Store unique global cluster Ids and count local number of contributions
    c.thresholdMask.resize(domain.nParticles());
    cstone::fillGpu(rawPtr(c.thresholdMask), rawPtr(c.thresholdMask)+domain.nParticles(), unsigned(0));
    thresholdMaskGpu(rawPtr(c.work_id)+domain.startIndex(), rawPtr(c.work_id)+domain.endIndex(),
        rawPtr(c.thresholdMask), unsigned(1));
    auto nGrouped = cstone::reduceGpu(rawPtr(c.thresholdMask), domain.nParticles(), size_t(0));
    h.nClusteredLocal = nGrouped;
    h.particleToHaloMap.resize(nGrouped);
    MPI_Allreduce(&h.nClusteredLocal, &h.nClusteredGlobal, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

    // Create map to sort particles by their cluster Ids
    flagSelectGpu(rawPtr(c.work_id)+domain.startIndex(), rawPtr(c.thresholdMask), rawPtr(c.idBuf),
        domain.nParticles(), rawPtr(c.keyBuf), domain.nParticlesWithHalos());
    cstone::sequenceGpu(rawPtr(tempBuf), domain.nParticles(), LocalIndex(domain.startIndex()));
    flagSelectGpu(rawPtr(tempBuf), rawPtr(c.thresholdMask), rawPtr(h.particleToHaloMap),
        domain.nParticles(), rawPtr(c.keyBuf), domain.nParticlesWithHalos());

    cstone::sortByKeyGpu(rawPtr(c.idBuf), rawPtr(c.idBuf)+nGrouped, rawPtr(h.particleToHaloMap));

    // Count local particle contributions
    auto nLocalClusters = cstone::uniqueCountGpu(rawPtr(c.idBuf), rawPtr(c.idBuf)+nGrouped);
    h.numClustersLocal = nLocalClusters;
    cstone::runLengthEncodeGpu(nGrouped, rawPtr(c.idBuf), rawPtr(c.idMap), rawPtr(tempBuf),
        rawPtr(c.numClusters), rawPtr(c.keyBuf), domain.nParticlesWithHalos());

    // Subtract 1 from unique cluster IDs to get zero-based indexing
    cstone::fillGpu(rawPtr(c.idBuf), rawPtr(c.idBuf)+nGrouped, ClusterIdType(1));
    cstone::subtractGpu(rawPtr(c.idMap), rawPtr(c.idBuf), nGrouped);

    // Compute local count and offset arrays
    cstone::fillGpu(rawPtr(h.localSize), rawPtr(h.localSize)+numClusters, uint32_t(0));
    cstone::scatterGpu(rawPtr(c.idMap), nLocalClusters, rawPtr(tempBuf), rawPtr(h.localSize));
    cstone::exclusiveScanGpu(rawPtr(h.localSize), rawPtr(h.localSize)+numClusters, rawPtr(h.localOffset));

    // Compute global counts and offset arrays
    convertUint32ToUint64Gpu(rawPtr(h.localSize), rawPtr(h.globalSize), numClusters);
    std::vector<uint64_t> globalCountHost(numClusters);
    std::vector<uint64_t> globalOffsetHost(numClusters);
    memcpyD2H(rawPtr(h.globalSize), numClusters, globalCountHost.data());
    MPI_Allreduce(MPI_IN_PLACE, globalCountHost.data(), numClusters, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    memcpyH2D(globalCountHost.data(), numClusters, rawPtr(h.globalSize));
    cstone::exclusiveScanGpu(rawPtr(h.globalSize), rawPtr(h.globalSize)+numClusters, rawPtr(h.globalOffset), uint64_t(0));

    cstone::sequenceGpu(rawPtr(h.cId), numClusters, ClusterIdType(1));
}
template void prepareParticleClusterMapGPU(
    cluster::ClusterData<cstone::GpuTag>& c,
    cluster::HaloData<cstone::GpuTag>& h,
    cstone::Domain<sph::SphTypes::KeyType, sph::SphTypes::CoordinateType, cstone::GpuTag>& domain
);

} // namespace cluster