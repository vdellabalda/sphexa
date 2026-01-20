/*! @file
 * @brief Friends of friends algorithm for halo finding
 *
 * This file implements the friends of friends (FoF) algorithm for halo finding in cosmological simulations.
 * The algorithm is designed to work with MPI and uses a tree-based approach to find halos in a distributed system.
 * 
 * @author Vincente Della Balda
 */

#include <queue>
#include <unordered_set>
#include <map>

#include "cstone/findneighbors.hpp"
#include "cluster_mpi_utils.hpp"
#include "union_find.hpp"

namespace cstone
{
/*! @brief traverse the octree finding all particles within a given distance
 *
 * @param domain              domain containing the octree, i.e. node data
 * @param i                   local particle index
 * @param clusterIdx          cluster ID to assign to the found particles
 * @param x                   coordinate input arrays
 * @param y    
 * @param z
 * @param percolationLength   linking length for the FoF algorithm
 * @param fofQueue            queue for found particles
 * @param particleClusterId   cluster ID array to assign the found particles to
 */
template<class DomainType,class Tc, class ClusterIdType, class KeyType>
ClusterIdType particleFOF(
     const LocalIndex i,
     const ClusterIdType clusterId,
     const Tc* x,
     const Tc* y,
     const Tc* z,
     const Tc percLength,
     const Tc percLengthSq,
     const OctreeNsView<Tc, KeyType>& tree,
     const Box<Tc>& box,
     std::queue<LocalIndex>& fofQueue,
     ClusterIdType* particleClusterId,
     const DomainType& domain
)
{     
     auto xi = x[i];
     auto yi = y[i];
     auto zi = z[i];
 
     Vec3<Tc> particle{xi, yi, zi};

     unsigned numNeighbors = 0;

     auto pbc    = BoundaryType::periodic;
     bool anyPbc = box.boundaryX() == pbc || box.boundaryY() == pbc || box.boundaryZ() == pbc;
     bool usePbc = anyPbc && !insideBox(particle, {percLength, percLength, percLength}, box);

     // Traverse the octree to find overlapping nodes
     auto overlapsPbc = [particle, percLengthSq,
          centers = tree.centers, sizes = tree.sizes, &box](TreeNodeIndex idx)
     { return norm2(minDistance(particle, centers[idx], sizes[idx], box)) < percLengthSq; };

     auto overlaps = [particle, percLengthSq,
          centers = tree.centers, sizes = tree.sizes](TreeNodeIndex idx)
     { return norm2(minDistance(particle, centers[idx], sizes[idx])) < percLengthSq; };
     
     // Brute force neighbour search within the leaf nodes
     auto searchBox=[particle, x, y, z, &tree, percLengthSq, clusterId, &particleClusterId,
          &fofQueue, &box, &numNeighbors](TreeNodeIndex idx)
     {     
          TreeNodeIndex leafIdx    = tree.internalToLeaf[idx];
          LocalIndex firstParticle = tree.layout[leafIdx];
          LocalIndex lastParticle  = tree.layout[leafIdx + 1];

          for (LocalIndex j = firstParticle; j < lastParticle; ++j)
          {
               if (particleClusterId[j]) continue;
               if (distanceSq<false>(
                    x[j], y[j], z[j], particle[0], particle[1], particle[2], box) < percLengthSq)
               {
                    particleClusterId[j] = clusterId;
                    fofQueue.push(j);
                    numNeighbors++;
               }
          }
     };

     auto searchBoxPbc=[particle, x, y, z, &tree, percLengthSq, clusterId, &particleClusterId,
          &fofQueue, &box, &numNeighbors](TreeNodeIndex idx)
     {     
          TreeNodeIndex leafIdx    = tree.internalToLeaf[idx];
          LocalIndex firstParticle = tree.layout[leafIdx];
          LocalIndex lastParticle  = tree.layout[leafIdx + 1];

          for (LocalIndex j = firstParticle; j < lastParticle; ++j)
          {
               if (particleClusterId[j]) continue;
               if (distanceSq<true>(
                    x[j], y[j], z[j], particle[0], particle[1], particle[2], box) < percLengthSq)
               {
                    particleClusterId[j] = clusterId;
                    fofQueue.push(j);
                    numNeighbors++;
               }
          }
     };

     if (usePbc) { singleTraversal(tree.childOffsets, tree.parents, overlapsPbc, searchBoxPbc); }
     else { singleTraversal(tree.childOffsets, tree.parents, overlaps, searchBox); }
     return numNeighbors;
}
} // namespace cstone

namespace cluster
{
using cstone::LocalIndex;


/*! @brief Friends of Friends algorithm for halo finding
 *
 * @param domain              the domain containing the particles and tree structure
 * @param globalClusterIdx    where to store global cluster ID of each particle
 * @param percolationLength   linking length for the FoF algorithm
 * @param x                   x-coordinates of the particles
 * @param y                   y-coordinates of the particles
 * @param z                   z-coordinates of the particles
 * @param myRank              MPI rank of the current process
 * @param numRanks            total number of MPI ranks
 */
template<class DomainType, class Box, class ParticleDataset, class ClusterDataset>
void computeLocalClusterIdImpl(
     size_t startIndex,
     size_t endIndex,
     ParticleDataset& d,
     ClusterDataset& c,
     Box& box,
     const int myRank,
     DomainType& domain
)
{
     const auto* x = d.x.data();
     const auto* y = d.y.data();
     const auto* z = d.z.data();

     std::fill(c.halo_id.data(), c.halo_id.data()+domain.nParticlesWithHalos(), unsigned(0));
     std::fill(c.flagged.data(), c.flagged.data()+domain.nParticlesWithHalos(), unsigned(0));
     std::fill(c.idBuf.data(), c.idBuf.data()+domain.nParticlesWithHalos(), unsigned(0));

     auto percolationLength = c.getPercLength();
     auto percolationLengthSq = percolationLength * percolationLength;

     std::queue<LocalIndex> fofQueue;
     LocalIndex currentSeed = 0;
     ClusterIdType currentClusterId = 0;
    
     for (size_t i = startIndex; i < endIndex; i++) {
          if (c.halo_id[i]) continue;
          currentClusterId += 1;
          c.halo_id[i] = currentClusterId;
          fofQueue.push(i);
          while (!(fofQueue.empty())) {
               currentSeed = fofQueue.front();
               fofQueue.pop();
               c.idBuf[currentClusterId-1] += 
                    cstone::particleFOF(
                         currentSeed,
                         currentClusterId,
                         x,
                         y,
                         z,
                         percolationLength,
                         percolationLengthSq,
                         d.treeView,
                         box,
                         fofQueue,
                         c.halo_id.data(),
                         domain
                    );
          };
     };

     // Assign global cluster keys
     assignClusterKey(
          c.halo_id.data(),
          c.localClusterKeys.data(),
          c.flagged.data(),
          domain.nParticlesWithHalos(),
          myRank
     );
     std::copy(c.localClusterKeys.data()+domain.startIndex(),
               c.localClusterKeys.data()+domain.endIndex(),
               c.globalClusterKeys.data()+domain.startIndex());
}


template<class ClusterDataset, class DomainType>
void computeGlobalClusterIdImpl(
     ClusterDataset& c,
     DomainType& domain,
     const int myRank
)
{
     LocalIndex nLocal = domain.nParticles();
     LocalIndex nHalo = domain.nParticlesWithHalos() - nLocal;
     LocalIndex nHaloStart = domain.startIndex();
     LocalIndex nHaloEnd = nHalo - nHaloStart; 
     size_t numClusteredHaloStart = std::accumulate(
          c.flagged.data(), c.flagged.data()+nHaloStart, size_t(0));
     size_t numClusteredHaloEnd = std::accumulate(
          c.flagged.data()+domain.endIndex(),c.flagged.data()+domain.nParticlesWithHalos(), size_t(0));        
 
     // edges for union-find
     c.edgeSrc.resize(numClusteredHaloStart + numClusteredHaloEnd);
     c.edgeDst.resize(numClusteredHaloStart + numClusteredHaloEnd);
     c.edges.resize(numClusteredHaloStart + numClusteredHaloEnd);
 
     int edgeIdx = 0;
     if (numClusteredHaloStart)
     {
          for (LocalIndex i=0; i<nHaloStart; ++i)
          {
               if (c.flagged[i])
               {
                    c.edgeSrc[edgeIdx] = c.localClusterKeys[i];
                    c.edgeDst[edgeIdx] = c.globalClusterKeys[i];
                    if (c.localClusterKeys[i] < c.globalClusterKeys[i]) {
                         c.edges[edgeIdx] = EdgeType{c.localClusterKeys[i], c.globalClusterKeys[i]};
                    }
                    else {
                         c.edges[edgeIdx] = EdgeType{c.globalClusterKeys[i], c.localClusterKeys[i]};
                    }
                    edgeIdx++;
               }
          }
     }
     if (numClusteredHaloEnd)
     {     
          for (LocalIndex i=domain.endIndex(); i<domain.nParticlesWithHalos(); ++i)
          {
               if (c.flagged[i])
               {
                    c.edgeSrc[edgeIdx] = c.localClusterKeys[i];
                    c.edgeDst[edgeIdx] = c.globalClusterKeys[i];
                    if (c.localClusterKeys[i] < c.globalClusterKeys[i]) {
                         c.edges[edgeIdx] = EdgeType{c.localClusterKeys[i], c.globalClusterKeys[i]};
                    } 
                    else {
                         c.edges[edgeIdx] = EdgeType{c.globalClusterKeys[i], c.localClusterKeys[i]};
                    }
                    edgeIdx++;
               }
          }
     }
 
     size_t numUniqueEdges = uniquify(c.edges.data(), edgeIdx);
 
     // Allgather Edges
     int numRanks;
     MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
     std::vector<int> recvCounts(numRanks);
     std::vector<int> displs(numRanks);
     int totalCount = allGathervSetup(
         numUniqueEdges,
         recvCounts.data(),
         displs.data()
     );
     std::vector<EdgeType> globalEdgesHost(totalCount);
     mpiAllgatherv(
         c.edges.data(), numUniqueEdges,
         globalEdgesHost.data(), recvCounts.data(), displs.data(),
         MPI_COMM_WORLD
     );
     
     // Compute unique cluster Ids from edges
     c.edgeSrc.resize(totalCount);
     c.edgeDst.resize(totalCount);
     
     for (int i=0; i<totalCount; ++i)
     {
         c.edgeSrc[i] = globalEdgesHost[i][0];
         c.edgeDst[i] = globalEdgesHost[i][1];
         c.keyBuf[i] = globalEdgesHost[i][0];
         c.keyBuf[i+totalCount] = globalEdgesHost[i][1];
     }
 
     size_t numUniqueEdgeKeys = uniquify(c.keyBuf.data(), 2*totalCount);
 
     c.clusterParents.resize(numUniqueEdgeKeys);
     c.clusterSizes.resize(numUniqueEdgeKeys);
     std::iota(c.clusterParents.begin(), c.clusterParents.end(), 0);
     std::fill(c.clusterSizes.begin(), c.clusterSizes.end(), 1);
     ClusterKeyType idx = 0;
     std::map<ClusterKeyType, ClusterKeyType> keyToGlobalId;
     for (size_t i=0; i<numUniqueEdgeKeys; ++i)
     {
          keyToGlobalId[c.keyBuf[i]] = idx++;
     }

     std::vector<EdgeType> edgeKeys(totalCount);
     for (int i=0; i<totalCount; ++i)
     {
          edgeKeys[i] = EdgeType{keyToGlobalId[c.edgeSrc[i]], keyToGlobalId[c.edgeDst[i]]};
     }

     unionFind(
          edgeKeys.data(),
          totalCount,
          c.clusterParents.data(),
          c.clusterSizes.data(),
          c.keyBuf.data()
     );

     // Update cluster Parents array to store root keys
     c.nonLocalKeys.resize(numUniqueEdgeKeys);
     for (size_t i=0; i<numUniqueEdgeKeys; ++i)
     {
          c.clusterParents[i] = findRoot(c.clusterParents.data(), i);
          c.nonLocalKeys[i] = c.keyBuf[c.clusterParents[i]];
     }

     // Update global cluster keys of local particles (only of halos which span MPI ranks)
     for (int i = domain.startIndex(); i < domain.endIndex(); ++i) 
     {
          // Check if particle is part of cluster spanning multiple ranks
          if (std::find(c.keyBuf.data(), c.keyBuf.data()+numUniqueEdgeKeys, c.localClusterKeys[i]) != c.keyBuf.data()+numUniqueEdgeKeys)
          {
               ClusterKeyType updatedClusterKey = c.keyBuf[c.clusterParents[keyToGlobalId[c.localClusterKeys[i]]]];
               c.localClusterKeys[i] = updatedClusterKey;
          }
     }

     // Update non-local keys list
     size_t numNonLocalKeys = uniquify(c.nonLocalKeys.data(), numUniqueEdgeKeys);
     c.nonLocalKeys.resize(numNonLocalKeys);
}


template<class ClusterKeyType>
size_t removeSmallClusters(
     ClusterKeyType* globalClusterIdx,
     ClusterIdType* finalClusterIdx,
     size_t nUniqueKeys,
     size_t memberThreshold,
     size_t startIndex,
     size_t endIndex
     )
{
     std::vector<size_t> localClusterMembers(nUniqueKeys, 0);
     std::vector<size_t> globalClusterMembers(nUniqueKeys, 0);
     std::vector<int> clusterMap(nUniqueKeys, 0);
     std::vector<ClusterIdType> reverseClusterMap(nUniqueKeys, 0);

     // Count members per cluster
	for (int pi=startIndex; pi < endIndex; ++pi)
     {
          if (globalClusterIdx[pi]!=0) 
          {
               localClusterMembers[globalClusterIdx[pi]-1]++;
          }
     }
     
     // Allreduce members per cluster
     MPI_Allreduce(localClusterMembers.data(), globalClusterMembers.data(), nUniqueKeys,
          MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
     
     // Evaluate how many clusters are above threshold
     size_t nClustersAboveThreshold = std::count_if(
          globalClusterMembers.begin(),
          globalClusterMembers.end(),
          [memberThreshold](size_t count){ return count>=memberThreshold; }
     );

     // Create a mapping from old cluster IDs to new cluster IDs
     std::iota(clusterMap.begin(), clusterMap.end(), 1);
     std::iota(reverseClusterMap.begin(), reverseClusterMap.end(), 1);
     cstone::sort_by_key(globalClusterMembers.begin(), globalClusterMembers.end(), clusterMap.data(), 
          std::greater<size_t>());
     cstone::sort_by_key(clusterMap.begin(), clusterMap.end(), reverseClusterMap.begin());

     // Clusters below threshold get mapped to 0
     std::replace_if(
          reverseClusterMap.begin(),
          reverseClusterMap.end(),
          [nClustersAboveThreshold](ClusterIdType clusterId){ return clusterId>nClustersAboveThreshold; },
          0
     );

     // Reassign cluster IDs
     for (size_t pi=startIndex; pi<endIndex; ++pi) {
          finalClusterIdx[pi] = (globalClusterIdx[pi]==0) ? 0 : reverseClusterMap[globalClusterIdx[pi]-1];
     }

     return nClustersAboveThreshold;
}


template<class ClusterDataset>
void computeCompactClusterIdImpl(
     ClusterDataset& c,
     LocalIndex startIndex,
     LocalIndex endIndex,
     const int myRank,
     const int numRanks
)
{    
     LocalIndex nParticles = endIndex - startIndex;

     c.uniqueKeys.resize(nParticles);
     c.keyCounts.resize(nParticles);
     std::fill(c.keyCounts.begin(), c.keyCounts.end(), 0);
     std::copy(c.localClusterKeys.data() + startIndex, c.localClusterKeys.data() + endIndex,
          c.keyBuf.data());

     size_t uniqueCount = runLengthEncode(c.keyBuf.data(), nParticles,
          c.uniqueKeys.data(), c.keyCounts.data());

     // Keys which appear less than cluster threshold times do not need to be communicated
     // unless they are part of a non-local cluster
     int keyCount = 0;
     c.uniqueKeys.resize(uniqueCount);
     c.keyCounts.resize(uniqueCount);
     std::unordered_set<ClusterKeyType> nonLocalKeySet(
          c.nonLocalKeys.data(),
          c.nonLocalKeys.data() + c.nonLocalKeys.size()
     );
     std::map<ClusterKeyType, ClusterKeyType> keyToGlobalId;
     for (int i = 0; i < uniqueCount; ++i)
     {
          if ((c.keyCounts[i] < c.getClusterThreshold()) && nonLocalKeySet.find(c.uniqueKeys[i]) == nonLocalKeySet.end())
               { 
                    keyToGlobalId[c.uniqueKeys[i]] = 0;
               }
          else 
               {
                    c.uniqueKeys[keyCount++] = c.uniqueKeys[i];
               }
     }
     c.uniqueKeys.resize(keyCount);

     // Gather all unique cluster keys across all ranks
     std::vector<int> recvCounts(numRanks), displs(numRanks);
     size_t totalCount = allGathervSetup(keyCount, recvCounts.data(), displs.data());
     if (totalCount > c.globalClusterKeys.size()) {
          c.globalClusterKeys.resize(totalCount);
     }
     mpiAllgatherv(c.uniqueKeys.data(), keyCount, c.globalClusterKeys.data(),
          recvCounts.data(), displs.data(), MPI_COMM_WORLD);
     size_t nUniqueKeys = uniquify(c.globalClusterKeys.data(), totalCount);

     ClusterKeyType idx = 1;
     for (size_t i=0; i<nUniqueKeys; ++i) {
          keyToGlobalId[c.globalClusterKeys[i]] = idx++;
     }
     // Assign global cluster IDs to local particles
     for (size_t i = startIndex; i < endIndex; ++i) {
          c.localClusterKeys[i] = keyToGlobalId[c.localClusterKeys[i]];          
     }
     // Remove clusters below threshold
     c.numClustersGlobal = removeSmallClusters(c.localClusterKeys.data(), c.halo_id.data(), nUniqueKeys,
          c.getClusterThreshold(), startIndex, endIndex);
}
} // namespace cluster