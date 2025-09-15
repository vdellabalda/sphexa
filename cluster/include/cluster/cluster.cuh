/*! @file
 * @brief Friends of friends algorithm for halo finding
 *
 * This file implements the friends of friends (FoF) algorithm for halo finding in cosmological simulations.
 * The algorithm is designed to work with MPI and uses a tree-based approach to find halos in a distributed system.
 * 
 * @author Vincente Della Balda
 */

#pragma once

#include "cstone/cuda/cub.hpp"
#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/primitives/warpscan.cuh"
#include "cstone/traversal/find_neighbors.cuh"
#include "cstone/traversal/groups_gpu.cuh"

namespace cstone

{

/*! @brief find octree node at depth n enclosing group of particles
 *
 * @tparam       Tc            float or double
 * @param[in]    pos_i         target body x,y,z,id
 * @param[in]    depth         depth of the octree node to find
 * 
 * @return                     octree node index at depth n enclosing group of particles
 */
template<class Tc, class KeyType>
__device__ KeyType warpBnode(const util::array<Vec4<Tc>, TravConfig::nwt>& pos_i,
                         const int depth)
{
    KeyType threadKeys[TravConfig::nwt];
    KeyType keyMins[TravConfig::nwt];
    KeyType keyMaxs[TravConfig::nwt];

    #pragma unroll
        for (int k = 0; k < TravConfig::nwt; ++k)
        {
            threadKeys[k] = hilbertKey(pos_i[k][0], pos_i[k][1], pos_i[k][2], maxTreeLevel<KeyType>{});
        }

    #pragma unroll
        for (int k=0; k < TravConfig::nwt; ++k)
        {
            keyMins[k] = warpMin(threadKeys[k]);
            keyMaxs[k] = warpMax(threadKeys[k]);
        }

    KeyType keyStart = min(keyMins[0], keyMins[1]);
    KeyType keyEnd   = max(keyMaxs[0], keyMaxs[1]);
    auto level = commonPrefix(keyStart, keyEnd) / 3;
    // This might cause issues later when comparing SFC neighbors at a preset depth.
    depth = min(level, depth);
    KeyType nodeKey = enclosingBoxCode(keyStart, depth);
    
    return nodeKey
}


/*! @brief count neighbors within a cutoff
 *
 * @tparam       Tc            float or double
 * @param[in]    sourceBody    source body x,y,z
 * @param[in]    validLaneMask number of lanes that contain valid source bodies
 * @param[in]    pos_i         target body x,y,z,id
 * @param[in]    box           global coordinate bounding box
 * @param[in]    targetBodyIdx index of target body of each lane
 * @param[in]    sourceBodyIdx index of source body of each lane
 * @param[inout] ec_i          target body edge counts to add to
 * @param[inout] eidx_i        target body indeces of edges
 * @param[inout] eidx_j        source body indeces of edges
 *
 * Number of computed particle-particle pairs per call is GpuConfig::warpSize^2 * TravConfig::nwt
 */
template<bool UsePbc, class Tc>
__device__ unsigned edgeCollector(
    Vec3<Tc> sourceBody,
    int numLanesValid,
    const util::array<Vec4<Tc>, TravConfig::nwt>& pos_i,
    const Box<Tc>& box,
    LocalIndex targetBodyIdx,
    LocalIndex sourceBodyIdx,
    unsigned ec_i,
    unsigned* eidx_i,
    unsigned* eidx_j) 
{   
    unsigned laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);
    bool neighbor[2];
    int localNeighborCount;
    int inclusiveShift;
    int shift;
    int writeIdx;
    int prefix;

    for (int j = 0; j < numLanesValid; j++)
    {
        Vec3<Tc> pos_j{shflSync(sourceBody[0], j), shflSync(sourceBody[1], j), shflSync(sourceBody[2], j)};
        cstone::LocalIndex idx_j = shflSync(sourceBodyIdx, j);
        localNeighborCount = 0;

#pragma unroll
        for (int k = 0; k < TravConfig::nwt; k++)
        {
            Tc d2 = distanceSq<UsePbc>(pos_j[0], pos_j[1], pos_j[2], pos_i[k][0], pos_i[k][1], pos_i[k][2], box);
            neighbor[k] = (d2 < pos_i[k][3]) && (d2 > Tc(0.0)) && (targetBodyIdx < sourceBodyIdx);
            localNeighborCount += neighbor[k] ? 1 : 0;
        }

    inclusiveShift = inclusiveScanInt(localNeighborCount);
    shift = inclusiveShift - localNeighborCount;
    writeIdx = ec_i + shift;

#pragma unroll
        for (int k = 0; k < TravConfig::nwt; k++)
        {
            if (neighbor[k])
            {   prefix = 0;
                for (int i = 0; i < k; i++) prefix += neighbor[i];
                eidx_i[writeIdx+prefix] = targetBodyIdx;
                eidx_j[writeIdx+prefix] = idx_j;
            }
        }
    ec_i += shflSync(inclusiveShift, GpuConfig::warpSize - 1);    
    }
    return ec_i;
}

__device__ int findRoot(int* clusterId, int node)
{
    int v = node;
    int u;
    int w;

    while (true)
    {   
        u = v;
        for (int i=0 ; i<2 ; i++)
        {
            v = clusterId[u];
            w = clusterId[v];
            atomicCAS(&clusterId[u], v, w);
        }
        if (v == w) return v;
    }
}

__device__ void unite(int* clusterId, int x, int y)
{
    int u = x;
    int v = y;
    while (true)
    {
        if (u == v) return;
        else if (u < v && atomicCAS(&clusterId[u], u, v) == u) return;
        else if (u > v && atomicCAS(&clusterId[v], v, u) == v) return;
        u = findRoot(clusterId, u);
        v = findRoot(clusterId, v);
    }
}


/*! @brief update cluster labels based on disjoint union set algorithm
*
* @param[in]    clusterId       cluster Label array
* @param[in]    eidx_i          edge indices i
* @param[in]    eidx_j          edge indices j
* @param[in]    numEdgesWarp    number of edges in the warp
*/
__device__ void partialDSU(int* clusterId, unsigned* eidx_i, unsigned* eidx_j, int numEdgesWarp)
{
    unsigned laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);
    int eid_i;
    int eid_j;
    int edgeId = 0;

while (numEdgesWarp > 0)
{
    if (numEdgesWarp >= GpuConfig::warpSize)
    {
        eid_i = eidx_i[edgeId + laneIdx];
        eid_j = eidx_j[edgeId + laneIdx];
        // union operation
        unite(clusterId, eid_i, eid_j);
        edgeId += TravConfig::targetSize;
        numEdgesWarp -= TravConfig::targetSize;
    }

    else
    {   
        const bool laneHasEdge = laneIdx < numEdgesWarp;
        eid_i = 
            laneHasEdge ? eidx_i[edgeId + laneIdx] : -1; 
        eid_j = 
            laneHasEdge ? eidx_j[edgeId + laneIdx] : -1;
        unite(clusterId, eid_i, eid_j);
        numEdgesWarp = 0;
    }
}
}

/*! @brief traverse one warp with up to TravConfig::targetSize target bodies down the tree
 *
 * @param[in]    eidx_i         
 * @param[in]    eidx_j
 * @param[in]    egmax
 * @param[in]    clusterId         
 * @param[in]    pos_i          target x,y,z,4h^2, TravConfig::nwt per lane
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
 * @return                      Number of P2P interactions tested to the group of target particles.
 *                              The total for the warp is the numbers returned here times the number of valid
 *                              targets in the warp.
 *
 * Constant input pointers are additionally marked __restrict__ to indicate to the compiler that loads
 * can be routed through the read-only/texture cache.
 */
template<bool UsePbc, class Tc, class Th, class KeyType, class Index>
__device__ uint2 traverseWarpDSU(unsigned* eidx_i,
                                 unsigned* eidx_j,
                                 unsigned egmax,
                                 int* clusterId,
                                 const util::array<Vec4<Tc>, TravConfig::nwt>& pos_i,
                                 const Vec3<Tc> targetCenter,
                                 const Vec3<Tc> targetSize,
                                 const Index targetBodyIdx[TravConfig::nwt],
                                 const Tc* __restrict__ x,
                                 const Tc* __restrict__ y,
                                 const Tc* __restrict__ z,
                                 const OctreeNsView<Tc, KeyType>& tree,
                                 int initNodeIdx,
                                 int depth,
                                 const Box<Tc>& box,
                                 volatile int* tempQueue,
                                 int* cellQueue)
{
    const TreeNodeIndex* __restrict__ childOffsets   = tree.childOffsets;
    const TreeNodeIndex* __restrict__ internalToLeaf = tree.internalToLeaf;
    const LocalIndex* __restrict__ layout            = tree.layout;
    const Vec3<Tc>* __restrict__ centers             = tree.centers;
    const Vec3<Tc>* __restrict__ sizes               = tree.sizes;

    const int laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);    

    unsigned p2pCounter = 0, maxStack = 0;

    int bodyQueue; // warp queue for source body indices

    // Compute starting node index
    //auto parentNodeIdx = warpBnode(pos_i, depth);
    //auto initNodeIdx = childOffsets[parentNodeIdx];
    // populate initial cell queue
    if (laneIdx == 0) { cellQueue[0] = initNodeIdx; }
    //if (laneIdx == 0) { cellQueue[0] = 1; }

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
                numEdgesWarp = edgeCollector<UsePbc>(sourceBody, GpuConfig::warpSize, pos_i, box, targetBodyIdx, bodyIdx, numEdgesWarp, eidx_i, eidx_j, radius2);
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
                    numEdgesWarp = edgeCollector<UsePbc>(sourceBody, GpuConfig::warpSize, pos_i, box, targetBodyIdx, bodyIdx, numEdgesWarp, eidx_i, eidx_j, radius2);
                    bdyFillLevel -= GpuConfig::warpSize;
                    // bodyQueue is now empty; put body indices that spilled into the queue
                    bodyQueue = shflDownSync(bodyIdx, numBodiesWarp - bdyFillLevel);
                    p2pCounter += GpuConfig::warpSize;
                }
                numBodiesWarp = 0; // No more bodies to process from current source cells
            }

            edgeMemFilled = numEdgesWarp + TravConfig::targetSize*GpuConfig::warpSize > egmax;
            if (edgeMemFilled) // If edge storage is full
            {
                partialDSU(clusterId, eidx_i, eidx_j, numEdgesWarp);
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
        const bool laneHasBody = laneIdx < bdyFillLevel;
        // Load position of source bodies, with padding for invalid lanes
        const Vec3<Tc> sourceBody =
            laneHasBody ? Vec3<Tc>{x[bodyQueue], y[bodyQueue], z[bodyQueue]} : Vec3<Tc>{Tc(0), Tc(0), Tc(0)};
            numEdgesWarp += edgeCollector<UsePbc>(sourceBody, bdyFillLevel, pos_i, box, targetBodyIdx, bodyQueue, numEdgesWarp, eidx_i, eidx_j, radius2);
        p2pCounter += bdyFillLevel;        
    }

    if (numEdgesWarp > 0) // If there are leftover edges
    {
        partialDSU(clusterId, eidx_i, eidx_j, numEdgesWarp);
    }

    return {p2pCounter, maxStack};
}

//! @brief edge search traversal statistics: sumP2P, maxP2P, maxStack
struct EcStats
{
    using type = unsigned long long;
    enum IndexNames
    {
        sumP2P,
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

template<class Tc, class Th, class Index>
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
        pos_i[i]      = {x[bodyIdx], y[bodyIdx], z[bodyIdx], radius};
    }
    return pos_i;
}


/*! @brief Find neighbors of a group of given particles, does not count self reference: min return value is 0
 *
 * @param[in]  bodyBegin   index of first particle in (x,y,z) to look for neighbors
 * @param[in]  bodyEnd     last (excluding) index of particle to look for neighbors
 * @param[in]  x           particle x coordinates
 * @param[in]  y           particle y coordinates
 * @param[in]  z           particle z coordinates
 * @param[in]  tree        octree connectivity and cell data
 * @param[in]  box         global coordinate bounding box
 * @param[in]  warpEidx    buffer for up to egmax edge indices for the (bodyEnd - bodyBegin) targets
 * @param[in]  egmax       maximum number of edge indices (target + source) storable in buffer
 * @param[in]  radius      cutoff radius for edge search
 * @param[inout]  clusterId   cluster Label array
 * @param[-]   globalPool  global memory for cell traversal stack
 * @return                 actual neighbor count of the particle handled by the executing warp lane, can be > ngmax,
 *
 * Note: Number of handled particles (bodyEnd - bodyBegin) should be GpuConfig::warpSize * TravConfig::nwt or smaller
 */
template<class Tc, class Th, class KeyType>
__device__ void traverseNeighborsDSU(cstone::LocalIndex bodyBegin,
                                                                       cstone::LocalIndex bodyEnd,
                                                                       const Tc* __restrict__ x,
                                                                       const Tc* __restrict__ y,
                                                                       const Tc* __restrict__ z,
                                                                       const OctreeNsView<Tc, KeyType>& tree,
                                                                       const Box<Tc>& box,
                                                                       cstone::LocalIndex* warpEidx,
                                                                       unsigned egmax,
                                                                       const Tc radius,
                                                                       int* clusterId,
                                                                       TreeNodeIndex* globalPool)
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

    uint2 warpStats;
    if (usePbc)
    {
        warpStats = traverseWarpDSU<true>(
            warpEidx, warpEidx+egmax, egmax, clusterId,
            pos_i, targetCenter, targetSize,
            x, y, z,
            tree, initNode, box, tempQueue, cellQueue);
    }
    else
    {
        warpStats = traverseWarpDSU<false>(
            warpEidx, warpEidx+egmax, egmax, clusterId,
            pos_i, targetCenter, targetSize,
            x, y, z,
            tree, initNode, box, tempQueue, cellQueue);
    }

    unsigned numP2P   = warpStats.x;
    unsigned maxStack = warpStats.y;
    assert(numP2P != 0xFFFFFFFF);

    if (laneIdx == 0)
    {
        unsigned targetGroupSize = bodyEnd - bodyBegin;
        atomicAdd(&ecStats[EcStats::sumP2P], EcStats::type(numP2P) * targetGroupSize);
        atomicMax(&ecStats[EcStats::maxP2P], EcStats::type(numP2P));
        atomicMax(&ecStats[EcStats::maxStack], EcStats::type(maxStack));
    }
}


template<class Tc, class T, class KeyType>
__global__ void clusterId_Gpu(
    unsigned egmax, const cstone::Box<Tc> box, const Tc radius,
    const LocalIndex* grpStart, const LocalIndex* grpEnd, LocalIndex numGroups,
    const cstone::OctreeNsView<Tc, KeyType> tree,
    const Tc* x, const Tc* y, const Tc* z,
    int* clusterId,
    LocalIndex* eidx, TreeNodeIndex* globalPool)
{
    unsigned laneIdx     = threadIdx.x & (GpuConfig::warpSize - 1);
    unsigned targetIdx   = 0;
    unsigned warpIdxGrid = (blockDim.x * blockIdx.x + threadIdx.x) >> GpuConfig::warpSizeLog2;

    LocalIndex* edgesWarp = eidx + egmax * warpIdxGrid;

    while (true)
    {
        // first thread in warp grabs next target
        if (laneIdx == 0) { targetIdx = atomicAdd(&cstone::targetCounterGlob, 1); }
        targetIdx = cstone::shflSync(targetIdx, 0);

        if (targetIdx >= numGroups) { break; }

        LocalIndex bodyBegin = grpStart[targetIdx];
        LocalIndex bodyEnd   = grpEnd[targetIdx];
        LocalIndex i         = bodyBegin + laneIdx;

        traverseNeighborsDSU(bodyBegin, bodyEnd, x, y, z, tree, box, edgesWarp, egmax, globalPool, radius, clusterId);
    }
}

template<bool avClean, class Dataset>
__host__ void computeClusterId(const GroupView& grp, Dataset& d, Dataset& c,
                      const cstone::Box<typename Dataset::RealType>& box)
{
    auto [traversalPool, eidxPool] = cstone::allocateNcStacks(d.devData.traversalStack, d.ngmax);
    unsigned egmax = d.ngmax * TravConfig::targetSize;

    cstone::resetEdgeTraversalCounters<<<1, 1>>>();

    clusterId_Gpu<<<TravConfig::numBlocks(), TravConfig::numThreads>>>(
        egmax, box, c.percolationLength,
        grp.groupStart, grp.groupEnd, grp.numGroups,
        d.treeView,
        rawPtr(d.devData.x), rawPtr(d.devData.y), rawPtr(d.devData.z),
        rawPtr(c.devData.halo_id),
        eidxPool, traversalPool);
    checkGpuErrors(cudaGetLastError());
}
}