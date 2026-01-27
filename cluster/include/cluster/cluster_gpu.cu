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

#include "definitions.h"

namespace cstone

{
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
__device__ unsigned edgeCollector(
    Vec3<Tc> sourceBody,
    int numLanesValid,
    const util::array<Vec4<Tc>, TravConfig::nwt>& pos_i,
    const Box<Tc>& box,
    const Index targetBodyIdx[TravConfig::nwt],
    LocalIndex sourceBodyIdx,
    unsigned ec_i,
    LocalIndex* eidx_i,
    LocalIndex* eidx_j) 
{   
    bool neighbor[TravConfig::nwt];
    int localNeighborCount;
    int inclusiveShift;
    int shift;
    int writeIdx;

    for (int j = 0; j < numLanesValid; j++)
    {   
        Vec3<Tc> pos_j{shflSync(sourceBody[0], j), shflSync(sourceBody[1], j), shflSync(sourceBody[2], j)};
        LocalIndex idx_j = shflSync(sourceBodyIdx, j);
    
        localNeighborCount = 0;

#pragma unroll
        for (int k = 0; k < TravConfig::nwt; k++)
        {   
            Tc d2 = distanceSq<UsePbc>(pos_j[0], pos_j[1], pos_j[2], pos_i[k][0], pos_i[k][1], pos_i[k][2], box);
            
            // Edges to halo particles are only found in one direction
            // Therefore the condition below is to too simple to avoid double counting
            // i.e. edges to halo particles with smaller index are not counted
            // neighbor[k] = (d2 < pos_i[k][3]) && (idx_j > targetBodyIdx[k]);

            neighbor[k] = (d2 < pos_i[k][3]);
            localNeighborCount += neighbor[k];
        }
    
    inclusiveShift = inclusiveScanInt(localNeighborCount);
    shift = inclusiveShift - localNeighborCount;
    writeIdx = ec_i + shift;
    
    int prefixSum = 0;
#pragma unroll
    for (int k = 0; k < TravConfig::nwt; k++)
        {
            if (neighbor[k])
            {
                eidx_i[writeIdx+prefixSum] = targetBodyIdx[k];
                eidx_j[writeIdx+prefixSum] = idx_j;
                prefixSum++;
            }
        }

    ec_i += shflSync(inclusiveShift, GpuConfig::warpSize - 1);

    }
    return ec_i;
}

// Find root with path compression
__device__ LocalIndex findRootGPU(LocalIndex* clusterId, LocalIndex node)
{
    LocalIndex v = node;
    LocalIndex u;
    LocalIndex w;

    while (true)
    {   
        u = v;
        //for (int i=0 ; i<2 ; i++)
        //{
        //    v = clusterId[u]; // read parent
        //    w = clusterId[v]; // read grandparent            
        //    atomicCAS(&clusterId[u], v, w); // path compression
        //}
        v = clusterId[u]; // read parent
        w = clusterId[v]; // read grandparent            
        atomicCAS(&clusterId[u], v, w); // path compression
        if (v == w) return v; // root found
    }
}


__device__ void uniteGPU(LocalIndex* clusterId, LocalIndex x, LocalIndex y, LocalIndex maxSize)
{
    if (x == y) return;
    
    LocalIndex u = x;
    LocalIndex v = y;    
    if (u >= maxSize || v >= maxSize) printf("Warning: Invalid cluster ID u=%u v=%u (max %u)\n", u, v, maxSize);
    while (true)
    {   
        u = findRootGPU(clusterId, u);
        v = findRootGPU(clusterId, v);

        if (u >= maxSize || v >= maxSize) printf("Warning: Invalid cluster ID u=%u v=%u (max %u)\n", u, v, maxSize);

        if (u == v) return;
        
        if (u > v) 
            {
                LocalIndex temp = u;
                u = v;
                v = temp;
            }
        atomicCAS(&clusterId[v], v, u);
    }
}


//struct updateRoot
//{
//    HOST_DEVICE_FUN 
//    void operator()(LocalIndex* clusterId, LocalIndex& node, LocalIndex& root)
//    {
//        root = findRoot(clusterId, node);
//    }
//};

/*! @brief update cluster labels based on disjoint union set algorithm
*
* @param[in]    clusterId       cluster Label array
* @param[in]    eidx_i          edge indices i
* @param[in]    eidx_j          edge indices j
* @param[in]    numEdgesWarp    number of edges in the warp
*/
__device__ void partialDSU(LocalIndex* clusterId, LocalIndex* eidx_i, LocalIndex* eidx_j, int numEdgesWarp, int numParticles,
    LocalIndex startIndex, LocalIndex endIndex, LocalIndex* changedHaloId)
{
    unsigned laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);
    LocalIndex eid_i;
    LocalIndex eid_j;
    int edgeId = 0;

while (numEdgesWarp > 0)
{
    const bool laneHasEdge = laneIdx < numEdgesWarp;

    if (numEdgesWarp >= GpuConfig::warpSize)
    {
        eid_i = eidx_i[edgeId + laneIdx];
        eid_j = eidx_j[edgeId + laneIdx];
    }

    else
    {   
        eid_i = 
            laneHasEdge ? eidx_i[edgeId + laneIdx] : 0; 
        eid_j =
            laneHasEdge ? eidx_j[edgeId + laneIdx] : 0;
    }

    if (laneHasEdge)
    {        
        // No need to do this atomically
        // Same adjustment for any thread and no back-and-forth changes
        changedHaloId[eid_i] = 1;
        changedHaloId[eid_j] = 1;
    }

    uniteGPU(clusterId, eid_i, eid_j, numParticles);
    edgeId += GpuConfig::warpSize;
    numEdgesWarp -= GpuConfig::warpSize;    
}
}

/*! @brief traverse one warp with up to TravConfig::targetSize target bodies down the tree
 *
 * @param[in]    eidx_i         buffer memory for target body indeces of edges        
 * @param[in]    eidx_j         buffer memory for source body indeces of edges
 * @param[in]    egmax          maximum number of edge indices (target + source) storable in buffer
 * @param[in]    clusterId      cluster Label array         
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
__device__ uint3 traverseWarpDSU(LocalIndex* eidx_i,
                                 LocalIndex* eidx_j,
                                 unsigned egmax,
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
                                 LocalIndex numParticles,
                                 LocalIndex startIndex,
                                 LocalIndex endIndex)
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

    int numEdgesWarp = 0;

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
        bool edgeMemFilled      = 0;
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
                numEdgesWarp = edgeCollector<UsePbc>(sourceBody, GpuConfig::warpSize, pos_i, box, targetBodyIdx, bodyIdx, numEdgesWarp, eidx_i, eidx_j);
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
                    numEdgesWarp = edgeCollector<UsePbc>(sourceBody, GpuConfig::warpSize, pos_i, box, targetBodyIdx, bodyQueue, numEdgesWarp, eidx_i, eidx_j);
                    bdyFillLevel -= GpuConfig::warpSize;
                    // bodyQueue is now empty; put body indices that spilled into the queue
                    bodyQueue = shflDownSync(bodyIdx, numBodiesWarp - bdyFillLevel);
                    p2pCounter += GpuConfig::warpSize;
                }
                numBodiesWarp = 0; // No more bodies to process from current source cells
            }

            edgeMemFilled = (numEdgesWarp + TravConfig::targetSize*GpuConfig::warpSize) > egmax;
            if (edgeMemFilled)
            {
                edgeCounter += numEdgesWarp;
                partialDSU(clusterId, eidx_i, eidx_j, numEdgesWarp, numParticles, startIndex, endIndex, changedHaloId);
                numEdgesWarp = 0;
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
            numEdgesWarp = edgeCollector<UsePbc>(sourceBody, bdyFillLevel, pos_i, box, targetBodyIdx, bodyQueue, numEdgesWarp, eidx_i, eidx_j);
            p2pCounter += bdyFillLevel;        
    }

    if (numEdgesWarp > 0) // If there are leftover edges
    {   
        edgeCounter += numEdgesWarp;
        partialDSU(clusterId, eidx_i, eidx_j, numEdgesWarp, numParticles, startIndex, endIndex, changedHaloId);
        numEdgesWarp = 0;
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
 * @param[in]  warpEidx    warp shared buffer for up to egmax edge indices
 * @param[in]  egmax       maximum number of edges storable in buffer
 * @param[in]  radius      cutoff radius for edge search
 * @param[inout]  clusterId   cluster Label array
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
                                     LocalIndex* warpEidx,
                                     unsigned egmax,
                                     const Tc radius,
                                     LocalIndex* clusterId,
                                     LocalIndex* changedClusterId,
                                     TreeNodeIndex* globalPool,
                                     LocalIndex numParticles,
                                     LocalIndex startIndex,
                                     LocalIndex endIndex)
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
            warpEidx, warpEidx+egmax, egmax, clusterId, changedClusterId,
            pos_i, targetCenter, targetSize, bodyIdx,
            x, y, z,
            tree, initNode, box, tempQueue, cellQueue, numParticles, startIndex, endIndex);
    }
    else
    {
        warpStats = traverseWarpDSU<false>(
            warpEidx, warpEidx+egmax, egmax, clusterId, changedClusterId,
            pos_i, targetCenter, targetSize, bodyIdx,
            x, y, z,
            tree, initNode, box, tempQueue, cellQueue, numParticles, startIndex, endIndex);
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
using cstone::GpuConfig;
using cstone::LocalIndex;
using cstone::TravConfig;
using cstone::TreeNodeIndex;

using util::FieldList;

template<class Tc, class KeyType>
__global__ void clusterIdGPU(
    unsigned egmax, const cstone::Box<Tc> box, const Tc radius,
    const LocalIndex* grpStart, const LocalIndex* grpEnd, LocalIndex numGroups,
    const cstone::OctreeNsView<Tc, KeyType> tree,
    const Tc* x, const Tc* y, const Tc* z,
    LocalIndex* clusterId,
    LocalIndex* changedClusterId,
    LocalIndex* eidx, TreeNodeIndex* globalPool,
    LocalIndex numParticles,
    LocalIndex startIndex,
    LocalIndex endIndex)
{

    const unsigned laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);
    const unsigned warpIdxGrid = (blockDim.x * blockIdx.x + threadIdx.x) >> GpuConfig::warpSizeLog2;

    int targetIdx             = 0;
    LocalIndex* edgesWarp = eidx +  2 * egmax * warpIdxGrid;

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
            edgesWarp, egmax, 
            radius, clusterId, changedClusterId,
            globalPool, numParticles, startIndex, endIndex);
    }
}

__global__ void updateRootGPU(LocalIndex* clusterId, LocalIndex lastBody)
{
    unsigned gid     = blockDim.x * blockIdx.x + threadIdx.x; 
    LocalIndex node  = gid;
    LocalIndex root;

    if (gid < lastBody)
    {
        root = cstone::findRootGPU(clusterId, node);
        clusterId[gid] = root;
    }
}

template<class LocalIndex>
__global__ void unionFindGpu(
    LocalIndex* clusterId,
    LocalIndex* edgeSrc,
    LocalIndex* edgeDst,
    size_t numEdges,
    size_t numClusters)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numEdges) return;

    cstone::uniteGPU(clusterId, edgeSrc[idx], edgeDst[idx], numClusters);
}


template<class ParticleDataset, class ClusterDataset>
__host__ void computeLocalClusterIdGPU(
    const cstone::GroupView& grp,
    ParticleDataset& d, ClusterDataset& c,
    const cstone::Box<typename ParticleDataset::RealType>& box,
    const int myRank)
{
    auto [traversalPool, eidxPool] = cstone::allocateNcStacks(d.traversalStack, d.ngmax);
    unsigned egmax = (d.ngmax * GpuConfig::warpSize) / 2;
    cstone::resetEdgeTraversalCounters<<<1, 1>>>();
    checkGpuErrors(cudaGetLastError());
    
    size_t numParticlesHalos = c.getNumParticlesHalos();
    auto percLength = c.getPercLength();
    c.devData.numClusters.resize(1);
    cstone::fillGpu(rawPtr(c.devData.flagged), rawPtr(c.devData.flagged)+numParticlesHalos, unsigned(0));
    cstone::sequenceGpu(rawPtr(c.devData.halo_id), numParticlesHalos, unsigned(0));

    clusterIdGPU<<<cstone::TravConfig::numBlocks(), TravConfig::numThreads>>>(
        egmax, box, percLength,
        grp.groupStart, grp.groupEnd, grp.numGroups,
        d.treeView,
        rawPtr(d.x), rawPtr(d.y), rawPtr(d.z),
        rawPtr(c.devData.halo_id), rawPtr(c.devData.flagged),
        eidxPool, traversalPool, numParticlesHalos, grp.firstBody, grp.lastBody);
    checkGpuErrors(cudaGetLastError());

    cstone::EcStats::type stats[cstone::EcStats::numStats];
    checkGpuErrors(cudaMemcpyFromSymbol(stats, GPU_SYMBOL(cstone::ecStats), cstone::EcStats::numStats * sizeof(cstone::EcStats::type)));

    cstone::EcStats::type maxP2P   = stats[cstone::EcStats::maxP2P];
    cstone::EcStats::type maxStack = stats[cstone::EcStats::maxStack];
    cstone::EcStats::type sumEdges = stats[cstone::EcStats::sumEdges];

    c.devData.edgesFoundEc = sumEdges;
    c.devData.stackUsedEc = maxStack;

    if (maxP2P == 0xFFFFFFFF) { throw std::runtime_error("GPU traversal stack exhausted in neighbor search\n"); }
    
    unsigned numThreads = 256;
    unsigned numBlocks  = (numParticlesHalos + numThreads - 1) / numThreads;
    if (numBlocks < 1) numBlocks = 1;
               
    updateRootGPU<<<numBlocks, numThreads>>>(rawPtr(c.devData.halo_id), numParticlesHalos);
    checkGpuErrors(cudaGetLastError());

    transformLocalToGlobalClusterKeys(
        rawPtr(c.devData.halo_id),
        rawPtr(c.devData.localClusterKeys),
        numParticlesHalos,
        myRank);
    
    checkGpuErrors(cudaGetLastError());

    memcpyD2D(rawPtr(c.devData.localClusterKeys)+grp.firstBody, grp.lastBody-grp.firstBody, rawPtr(c.devData.globalClusterKeys)+grp.firstBody);
    
    memcpyD2D(rawPtr(c.devData.localClusterKeys)+grp.firstBody, grp.lastBody-grp.firstBody, rawPtr(c.devData.keyBuf));
    cstone::sortGpu(rawPtr(c.devData.keyBuf), rawPtr(c.devData.keyBuf)+grp.lastBody - grp.firstBody, rawPtr(c.devData.globalClusterKeys));
    size_t numUniqueKeys = cstone::uniqueCountGpu(
        rawPtr(c.devData.keyBuf),
        rawPtr(c.devData.keyBuf)+grp.lastBody - grp.firstBody);

    memcpyD2D(rawPtr(c.devData.localClusterKeys)+grp.firstBody, grp.lastBody-grp.firstBody, rawPtr(c.devData.globalClusterKeys)+grp.firstBody);
    return;
}

template void computeLocalClusterIdGPU(
    const cstone::GroupView& grp,
    sphexa::ParticlesData<cstone::GpuTag>& d,
    cluster::ClusterData<cstone::GpuTag>& c,
    const cstone::Box<sph::SphTypes::CoordinateType>& box,
    const int myRank);

template<class KeyType, class IndexType>
__global__ void clusterKeyUpdate(
    KeyType* localKeys,
    IndexType* clusterIds,
    IndexType* selectFlags,
    const IndexType* clusterParents,
    const KeyType* uniqueKeys,
    size_t numParticles)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= numParticles) return;
    if (selectFlags[tid] == 0) return;

    IndexType localId = clusterIds[tid];
    KeyType oldKey = localKeys[tid];
    IndexType newClusterId = clusterParents[localId];
    KeyType newKey = uniqueKeys[newClusterId];
    localKeys[tid] = newKey;

    return;
}

template<class KeyType, class EdgeType>
__global__ void constructEdgesGpu(KeyType* edgeSrc, KeyType* edgeDst, size_t n, EdgeType* edges)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {return;}

    if (edgeSrc[idx] < edgeDst[idx])
    {
        edges[idx][0] = edgeSrc[idx];
        edges[idx][1] = edgeDst[idx];
    }
    else
    {
        edges[idx][0] = edgeDst[idx];
        edges[idx][1] = edgeSrc[idx];
    }
}

template<class EdgeType, class KeyType>
__global__ void flattenEdgesGpu(EdgeType* edges, size_t n, KeyType* edgeIds)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {return;}

    edgeIds[idx]   = edges[idx][0];
    edgeIds[idx+n] = edges[idx][1];
}


template<class ClusterIdType, class FlagType>
__global__ void flagNonLocalKeysGpu(
    const ClusterIdType* clusterParents,
    FlagType* rootFlags,
    size_t numUniqueKeys
)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numUniqueKeys) {return;}
    
    if (idx == clusterParents[idx])
    {
        rootFlags[idx] = 1;
    }
}

template<class ClusterKeyType, class IdType>
__global__ void computeClusterOwnershipGpu(
    const ClusterKeyType* uniqueKeys,
    IdType* ownership,
    size_t numUniqueKeys
)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numUniqueKeys) {return;}

    ownership[idx] = getRankFromClusterKey(uniqueKeys[idx]);
}

// To do:
// - exchange compact copy with cub::DeviceSelect::Flagged

template<class ParticleDataset, class ClusterDataset, class DomainType>
__host__ void computeGlobalClusterIdGPU(
    ParticleDataset& d, ClusterDataset& c, DomainType& domain, const int myRank
)
{   
    ClusterKeyHashMapManager<ClusterKeyType, ClusterIdType> hashMapManager;
    cstone::DeviceVector<int> tempNumSelected(1);
    unsigned numThreads = 256;

    LocalIndex nLocal = domain.nParticles();
    LocalIndex nHalo = domain.nParticlesWithHalos() - nLocal;
    LocalIndex nHaloStart = domain.startIndex();
    LocalIndex nHaloEnd = nHalo - nHaloStart;

    size_t numClusteredHaloStart = cstone::reduceGpu(rawPtr(c.devData.flagged), nHaloStart, size_t(0));
    size_t numClusteredHaloEnd = cstone::reduceGpu(rawPtr(c.devData.flagged)+nHaloStart+nLocal, nHaloEnd, size_t(0));        
    
    // edges for union-find
    c.devData.edgeSrc.resize(numClusteredHaloStart + numClusteredHaloEnd);
    c.devData.edgeDst.resize(numClusteredHaloStart + numClusteredHaloEnd);
    c.devData.edges.resize(numClusteredHaloStart + numClusteredHaloEnd);

    if (numClusteredHaloStart)
    {
        flagSelectGpu(
            rawPtr(c.devData.globalClusterKeys),
            rawPtr(c.devData.flagged),
            rawPtr(c.devData.edgeSrc),
            rawPtr(tempNumSelected),
            nHaloStart
        );
        flagSelectGpu(
            rawPtr(c.devData.localClusterKeys),
            rawPtr(c.devData.flagged),
            rawPtr(c.devData.edgeDst),
            rawPtr(tempNumSelected),
            nHaloStart
        );

    }
    if (numClusteredHaloEnd)
    {
        flagSelectGpu(
            rawPtr(c.devData.globalClusterKeys)+domain.endIndex(),
            rawPtr(c.devData.flagged)+domain.endIndex(),
            rawPtr(c.devData.edgeSrc)+numClusteredHaloStart,
            rawPtr(tempNumSelected),
            nHaloEnd
        );
        flagSelectGpu(
            rawPtr(c.devData.localClusterKeys)+domain.endIndex(),
            rawPtr(c.devData.flagged)+domain.endIndex(),
            rawPtr(c.devData.edgeDst)+numClusteredHaloStart,
            rawPtr(tempNumSelected),
            nHaloEnd
        );
    }

    unsigned numBlocks  = (numClusteredHaloStart + numClusteredHaloEnd + numThreads - 1) / numThreads;
    if (numBlocks < 1) numBlocks = 1;
    constructEdgesGpu<<<numBlocks, numThreads>>>(rawPtr(c.devData.edgeSrc), rawPtr(c.devData.edgeDst), numClusteredHaloStart+numClusteredHaloEnd, rawPtr(c.devData.edges));
    checkGpuErrors(cudaGetLastError());  

    cstone::sortGpu(rawPtr(c.devData.edges), rawPtr(c.devData.edges)+numClusteredHaloStart+numClusteredHaloEnd);    
    EdgeType* newEdgesEnd = cstone::uniqueGpu(rawPtr(c.devData.edges), rawPtr(c.devData.edges)+numClusteredHaloStart+numClusteredHaloEnd);
    size_t numUniqueEdges = newEdgesEnd-rawPtr(c.devData.edges);

    // Allgather of edges
    int numRanks;
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    std::vector<int> recvCounts(numRanks);
    std::vector<int> displs(numRanks), counts(numRanks);
    std::fill(counts.begin(), counts.end(), 1);
    std::iota(displs.begin(), displs.end(), 0);

    mpiAllgatherv(&numUniqueEdges, 1, recvCounts.data(), counts.data(), displs.data(), MPI_COMM_WORLD);

    displs[0] = 0;
    for (int i = 1; i < numRanks; ++i)
        displs[i] = displs[i - 1] + recvCounts[i - 1];
    int totalCount = displs.back() + recvCounts.back();

    std::vector<EdgeType> globalEdgesHost(totalCount);
    std::vector<EdgeType> localEdgesHost(numUniqueEdges);
    memcpyD2H(rawPtr(c.devData.edges), numUniqueEdges, localEdgesHost.data());
    
    mpiAllgatherv(
        localEdgesHost.data(), numUniqueEdges,
        globalEdgesHost.data(), recvCounts.data(), displs.data(),
        MPI_COMM_WORLD
    );
    
    c.devData.edges.resize(totalCount);
    memcpyH2D(globalEdgesHost.data(), totalCount, rawPtr(c.devData.edges));
    // Allgather complete

    // Compute unique cluster Ids from edges
    c.devData.edgeSrc.resize(totalCount);
    c.devData.edgeDst.resize(totalCount);
    numBlocks = (totalCount + numThreads - 1) / numThreads;
    if (numBlocks < 1) numBlocks = 1;
    flattenEdgesGpu<<<numBlocks, numThreads>>>(
        rawPtr(c.devData.edges),
        totalCount,
        rawPtr(c.devData.keyBuf)
    );
    checkGpuErrors(cudaGetLastError());

    memcpyD2D(rawPtr(c.devData.keyBuf), totalCount, rawPtr(c.devData.edgeSrc));
    memcpyD2D(rawPtr(c.devData.keyBuf)+totalCount, totalCount, rawPtr(c.devData.edgeDst));

    cstone::sortGpu(rawPtr(c.devData.keyBuf), rawPtr(c.devData.keyBuf)+2*totalCount);
    ClusterKeyType* newKeysEnd = cstone::uniqueGpu(rawPtr(c.devData.keyBuf), rawPtr(c.devData.keyBuf)+2*totalCount);
    size_t numUniqueEdgeKeys = newKeysEnd - rawPtr(c.devData.keyBuf);
    
    cstone::sequenceGpu(rawPtr(c.devData.idBuf), numUniqueEdgeKeys, unsigned(0)); // reset idBuf to sequence 0,1,2,...
    checkGpuErrors(cudaGetLastError());
    checkGpuErrors(hashMapManager.initialize(rawPtr(c.devData.keyBuf), rawPtr(c.devData.idBuf), numUniqueEdgeKeys));
    
    // Convert Edges to Edge Indices
    checkGpuErrors(hashMapManager.lookupBatch(
        rawPtr(c.devData.edgeSrc),
        rawPtr(c.devData.idBuf),
        totalCount
    ));
    checkGpuErrors(hashMapManager.lookupBatch(
        rawPtr(c.devData.edgeDst),
        rawPtr(c.devData.idBuf)+totalCount,
        totalCount
    ));

    // perform union-find on edges
    c.devData.clusterParents.resize(numUniqueEdgeKeys);
    cstone::sequenceGpu(rawPtr(c.devData.clusterParents), numUniqueEdgeKeys, unsigned(0));
    checkGpuErrors(cudaGetLastError());

    numBlocks  = (totalCount + numThreads - 1) / numThreads;
    if (numBlocks < 1) numBlocks = 1;                    
    unionFindGpu<<<numBlocks, numThreads>>>(
        rawPtr(c.devData.clusterParents),
        rawPtr(c.devData.idBuf),
        rawPtr(c.devData.idBuf)+totalCount,
        totalCount,
        numUniqueEdgeKeys
    );
    checkGpuErrors(cudaGetLastError());    
    updateRootGPU<<<numBlocks, numThreads>>>(rawPtr(c.devData.clusterParents), numUniqueEdgeKeys);
    checkGpuErrors(cudaGetLastError());
 
    // Update local cluster keys to global cluster keys
    checkGpuErrors(hashMapManager.lookupBatchFlag(
        rawPtr(c.devData.localClusterKeys)+domain.startIndex(),
        rawPtr(c.devData.idBuf)+domain.startIndex(),
        rawPtr(c.devData.halo_id)+domain.startIndex(),
        domain.nParticles())
    );
    numBlocks  = (domain.nParticles() + numThreads - 1) / numThreads;
    clusterKeyUpdate<<<numBlocks, numThreads>>>(
        rawPtr(c.devData.localClusterKeys)+domain.startIndex(),
        rawPtr(c.devData.idBuf)+domain.startIndex(),
        rawPtr(c.devData.halo_id)+domain.startIndex(),
        rawPtr(c.devData.clusterParents),
        rawPtr(c.devData.keyBuf),
        domain.nParticles()
    );    
    checkGpuErrors(cudaGetLastError());

    // Roots of the union-find are all (global) non-pure clusters
    c.devData.thresholdMask.resize(numUniqueEdgeKeys);
    c.devData.nonLocalKeys.resize(numUniqueEdgeKeys);
    cstone::fillGpu(rawPtr(c.devData.thresholdMask), rawPtr(c.devData.thresholdMask)+numUniqueEdgeKeys, unsigned(0));
    numBlocks = (numUniqueEdgeKeys + numThreads - 1) / numThreads;
    if (numBlocks < 1) numBlocks = 1;
    flagNonLocalKeysGpu<<<numBlocks, numThreads>>>(
        rawPtr(c.devData.clusterParents),
        rawPtr(c.devData.thresholdMask),
        numUniqueEdgeKeys
    );
    flagSelectGpu(
        rawPtr(c.devData.keyBuf),
        rawPtr(c.devData.thresholdMask),
        rawPtr(c.devData.nonLocalKeys),
        rawPtr(tempNumSelected),
        numUniqueEdgeKeys
    );

    return;
}

template void computeGlobalClusterIdGPU(
    sphexa::ParticlesData<cstone::GpuTag>& d,
    cluster::ClusterData<cstone::GpuTag>& c,
    cstone::Domain<sph::SphTypes::KeyType, sph::SphTypes::CoordinateType, cstone::GpuTag>& domain,
    const int myRank);


// Cluster remapping with binary search
template<class ClusterKeyType, class ClusterIdType>
__global__ void directClusterRemapping(
    const ClusterKeyType* localClusterKeys,
    ClusterIdType* finalClusterIds,
    const ClusterIdType* selectFlags,
    const ClusterKeyType* sortedKeys,
    const ClusterIdType* sortedIds,
    size_t numParticles,
    size_t numSortedKeys)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numParticles) return;
    if (selectFlags[tid] == 0) 
        {
            finalClusterIds[tid] = 0;
            return;
        }

    ClusterKeyType myKey = localClusterKeys[tid];
    size_t left = 0, right = numSortedKeys;
    bool found = false;
    ClusterIdType compactId = 0;
    size_t mid;
    ClusterKeyType midKey;
    
    while (left < right) {
        mid = (left + right) / 2;
        midKey = sortedKeys[mid];
        
        if (midKey == myKey) {
            compactId = sortedIds[mid];
            found = true;
            break;
        } else if (midKey < myKey) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    finalClusterIds[tid] = compactId;
}

template<class ClusterDataset, class DomainType>
__host__ void computeCompactClusterIdGPU(
    ClusterDataset& c,
    DomainType& domain,
    const int myRank, const int numRanks)
{   
    
    cstone::DeviceVector<int> tempNumSelected(1);
    flagSelectGpu(
            rawPtr(c.devData.localClusterKeys)+domain.startIndex(),
            rawPtr(c.devData.flagged)+domain.startIndex(),
            rawPtr(c.devData.globalClusterKeys),
            rawPtr(tempNumSelected),
            domain.nParticles()
    );

    size_t numClustered = cstone::reduceGpu(rawPtr(c.devData.flagged)+domain.startIndex(), domain.nParticles(), size_t(0));
    cstone::sortGpu(rawPtr(c.devData.globalClusterKeys), rawPtr(c.devData.globalClusterKeys)+numClustered, rawPtr(c.devData.keyBuf));
    checkGpuErrors(cudaGetLastError());
    auto numUniqueKeys = cstone::uniqueCountGpu(rawPtr(c.devData.globalClusterKeys), rawPtr(c.devData.globalClusterKeys)+numClustered);
    checkGpuErrors(cudaGetLastError());
    c.devData.uniqueKeys.resize(numUniqueKeys); 
    c.devData.keyCounts.resize(numUniqueKeys);
    cstone::runLengthEncodeGpu(numClustered, rawPtr(c.devData.globalClusterKeys), rawPtr(c.devData.uniqueKeys), rawPtr(c.devData.keyCounts), rawPtr(c.devData.numClusters));
    checkGpuErrors(cudaGetLastError());

    // Find intersection between local clusters and all (global) non-pure clusters
    // To find local non-pure clusters
    size_t numNonLocalKeys = cstone::reduceGpu(rawPtr(c.devData.thresholdMask), c.devData.thresholdMask.size(), size_t(0));
    std::pair<ClusterKeyType*, ClusterIdType*> newNonLocalIterators = setIntersectionByKeyGpu(
        rawPtr(c.devData.uniqueKeys), numUniqueKeys,
        rawPtr(c.devData.nonLocalKeys), numNonLocalKeys,
        rawPtr(c.devData.keyCounts),
        rawPtr(c.devData.keyBuf),
        rawPtr(c.devData.idBuf)
    );
    size_t nonLocalCount = newNonLocalIterators.first - rawPtr(c.devData.keyBuf);

    // Find set difference between local pure and non-pure clusters
    // To find local pure clusters
    std::pair<ClusterKeyType*, ClusterIdType*> newLocalIterators = setDifferenceByKeyGpu(
        rawPtr(c.devData.uniqueKeys), numUniqueKeys,
        rawPtr(c.devData.keyBuf), nonLocalCount,
        rawPtr(c.devData.keyCounts),
        rawPtr(c.devData.idBuf),
        rawPtr(c.devData.keyBuf)+nonLocalCount,
        rawPtr(c.devData.idBuf)+nonLocalCount
    );
    size_t localCount = newLocalIterators.first - (rawPtr(c.devData.keyBuf)+nonLocalCount);
    
    // Mark local clusters above threshold for sending
    c.devData.thresholdMask.resize(nonLocalCount+localCount);
    cstone::fillGpu(rawPtr(c.devData.thresholdMask), rawPtr(c.devData.thresholdMask)+nonLocalCount, unsigned(1));
    thresholdMaskGpu(
        rawPtr(c.devData.idBuf)+nonLocalCount,
        rawPtr(c.devData.idBuf)+nonLocalCount+localCount,
        rawPtr(c.devData.thresholdMask)+nonLocalCount,
        c.getClusterThreshold()
    );

    //size_t sendCount = cstone::reduceGpu(rawPtr(c.devData.thresholdMask), nonLocalCount+localCount, size_t(0));
    
    flagSelectGpu(
        rawPtr(c.devData.keyBuf),
        rawPtr(c.devData.thresholdMask),
        rawPtr(c.devData.uniqueKeys),
        rawPtr(tempNumSelected),
        nonLocalCount+localCount
    );
    flagSelectGpu(
        rawPtr(c.devData.idBuf),
        rawPtr(c.devData.thresholdMask),
        rawPtr(c.devData.keyCounts),
        rawPtr(tempNumSelected),
        nonLocalCount+localCount
    );

    
    // Allgather Keys
    size_t sendCount = cstone::reduceGpu(rawPtr(c.devData.thresholdMask), nonLocalCount+localCount, size_t(0));
    std::vector<int> recvCounts(numRanks);
    std::vector<int> displs(numRanks), counts(numRanks);
    std::fill(counts.begin(), counts.end(), 1);
    std::iota(displs.begin(), displs.end(), 0);

    mpiAllgatherv(&sendCount, 1, recvCounts.data(), counts.data(), displs.data(), MPI_COMM_WORLD);

    displs[0] = 0;
    for (int i = 1; i < numRanks; ++i)
        displs[i] = displs[i - 1] + recvCounts[i - 1];
    int totalCount = displs.back() + recvCounts.back();

    std::vector<ClusterKeyType> globalKeysHost(totalCount);
    std::vector<ClusterIdType> globalKeyCountsHost(totalCount);
    std::vector<ClusterKeyType> localKeysHost(sendCount);
    std::vector<ClusterIdType> localKeyCountsHost(sendCount);
    memcpyD2H(rawPtr(c.devData.uniqueKeys), sendCount, localKeysHost.data());
    
    mpiAllgatherv(
        localKeysHost.data(), sendCount,
        globalKeysHost.data(), recvCounts.data(), displs.data(),
        MPI_COMM_WORLD
    );
    
    memcpyH2D(globalKeysHost.data(), totalCount, rawPtr(c.devData.globalClusterKeys));    
    // Allgather key counts
    memcpyD2H(rawPtr(c.devData.keyCounts), sendCount, localKeyCountsHost.data());
    mpiAllgatherv(
        localKeyCountsHost.data(), sendCount,
        globalKeyCountsHost.data(), recvCounts.data(), displs.data(),
        MPI_COMM_WORLD
    );
    memcpyH2D(globalKeyCountsHost.data(), totalCount, rawPtr(c.devData.idBuf));
    // Allgather complete

    cstone::sortByKeyGpu(
        rawPtr(c.devData.globalClusterKeys), rawPtr(c.devData.globalClusterKeys)+totalCount,
        rawPtr(c.devData.idBuf),
        rawPtr(c.devData.globalClusterKeys)+totalCount,
        rawPtr(c.devData.idBuf)+totalCount,
        rawPtr(c.devData.keyBuf),
        domain.nParticlesWithHalos()
    );

    numUniqueKeys = cstone::uniqueCountGpu(rawPtr(c.devData.globalClusterKeys), rawPtr(c.devData.globalClusterKeys)+totalCount);
    
    c.devData.uniqueKeys.resize(numUniqueKeys);
    c.devData.keyCounts.resize(numUniqueKeys);
    c.devData.thresholdMask.resize(numUniqueKeys);
    c.devData.idMap.resize(numUniqueKeys);

    std::pair<ClusterKeyType*, ClusterIdType*> new_iterators = cstone::reduceByKeyGpu(
        rawPtr(c.devData.globalClusterKeys), rawPtr(c.devData.globalClusterKeys)+totalCount,
        rawPtr(c.devData.idBuf),
        rawPtr(c.devData.uniqueKeys),
        rawPtr(c.devData.keyCounts)
    );
    
    cstone::sortByKeyDescendGpu(
        rawPtr(c.devData.keyCounts), rawPtr(c.devData.keyCounts)+numUniqueKeys,
        rawPtr(c.devData.uniqueKeys)
    );

    thresholdMaskGpu(rawPtr(c.devData.keyCounts), rawPtr(c.devData.keyCounts)+numUniqueKeys, rawPtr(c.devData.thresholdMask), c.getClusterThreshold());
    cstone::inclusiveScanGpu(rawPtr(c.devData.thresholdMask), rawPtr(c.devData.thresholdMask)+numUniqueKeys, rawPtr(c.devData.idMap));
    cstone::multiplyElementWiseGpu(rawPtr(c.devData.idMap), rawPtr(c.devData.idMap)+numUniqueKeys, rawPtr(c.devData.thresholdMask));

    cstone::sortByKeyGpu(
        rawPtr(c.devData.uniqueKeys), rawPtr(c.devData.uniqueKeys)+numUniqueKeys,
        rawPtr(c.devData.idMap),
        rawPtr(c.devData.keyBuf),
        rawPtr(c.devData.idBuf),
        rawPtr(c.devData.globalClusterKeys),
        domain.nParticlesWithHalos()
    );

    // Use direct binary search:
    unsigned numThreads = 256;
    unsigned numBlocks = (domain.nParticles() + numThreads - 1) / numThreads;
    if (numBlocks < 1) numBlocks = 1;
    directClusterRemapping<<<numBlocks, numThreads>>>(
        rawPtr(c.devData.localClusterKeys)+domain.startIndex(),
        rawPtr(c.devData.halo_id)+domain.startIndex(),
        rawPtr(c.devData.flagged)+domain.startIndex(),
        rawPtr(c.devData.uniqueKeys), // already sorted
        rawPtr(c.devData.idMap),      // corresponding compact IDs
        domain.nParticles(),
        numUniqueKeys
    );
    checkGpuErrors(cudaGetLastError());

    // Remap non-local clusters to compact Ids
    cstone::fillGpu(rawPtr(c.devData.idBuf), rawPtr(c.devData.idBuf)+numNonLocalKeys, ClusterIdType(1));
    numBlocks = (numNonLocalKeys + numThreads - 1) / numThreads;
    if (numBlocks < 1) numBlocks = 1;
    directClusterRemapping<<numBlocks, numThreads>>>(
        rawPtr(c.devData.nonLocalKeys),
        rawPtr(c.devData.nonLocalKeys),
        rawPtr(c.devData.thresholdMask),
        rawPtr(c.devData.uniqueKeys), // already sorted
        rawPtr(c.devData.idMap),      // corresponding compact IDs
        numNonLocalKeys,
        numUniqueKeys
    );


}
template void computeCompactClusterIdGPU(
    cluster::ClusterData<cstone::GpuTag>& c,
    cstone::Domain<sph::SphTypes::KeyType, sph::SphTypes::CoordinateType, cstone::GpuTag>& domain,
    const int myRank, const int numRanks);
}
