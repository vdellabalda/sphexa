/*! @file
 * @brief Friends of friends algorithm for halo finding
 *
 * This file implements the friends of friends (FoF) algorithm for halo finding in cosmological simulations.
 * The algorithm is designed to work with MPI and uses a tree-based approach to find halos in a distributed system.
 * 
 * @author Vincente Della Balda
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <queue>
#include <algorithm>
#include <set>
#include <map>
#include <mpi.h>

#include "cluster.hpp"
#include "cstone/findneighbors.hpp"

using namespace cstone;

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
template<class KeyType, class Tc, class T, class ClusterIdType, class Accelerator= CpuTag>
unsigned particleFOF(
     Domain<KeyType, T, Accelerator>& domain,
     const LocalIndex i,
     const ClusterIdType clusterIdx,
     const Tc* x,
     const Tc* y,
     const Tc* z,
     const Tc percolationLength,
     std::queue<int>& fofQueue,
     std::vector<ClusterIdType>& particleClusterId,
     int myRank
)
{
     TreeNodeIndex startCell = domain.startCell();
     TreeNodeIndex endCell = domain.endCell();
     int startIndex = domain.startIndex();
     int endIndex = domain.endIndex();

     auto internalToLeaf = domain.focusTree().octreeViewAcc().internalToLeaf;
     auto particleOffsets = domain.layout().data();
     auto childOffsets = domain.focusTree().octreeViewAcc().childOffsets;
     auto leafCounts = domain.focusTree().leafCounts().data();
     auto centers = domain.focusTree().geoCentersAcc().data();
     auto sizes = domain.focusTree().geoSizesAcc().data();

     auto xi = x[i];
     auto yi = y[i];
     auto zi = z[i];
 
     Vec3<Tc> particle{xi, yi, zi};

     unsigned numNeighbors = 0;
 
     /*
     Look into leaf nodes which are close enough
     There are two scenarios: Particle is contained in node or outside
     If contained, obviously we should expand the node
     If outside, we check whether the distance between the particle and the node surface is smaller than the percolation length
     
     Not yet implemented:
     - Periodicity
 
     Thoughts:
     - If a node is fully contained in the ball around particle i, all the particles within the node should be added to the cluster,
     without need to traverse the tree further or perform any particle-particle distance calculations.
     Is it possible to exit the traversal loop prematurely? Also, this requires a more complex particle-node distance calculation.
     Even more so, if the particles in the node are already part of another cluster, they should be skipped.
     Though if a particle is already part of a cluster, then all other particles in the same node would be part of the same cluster too.
     It would therefore be beneficial to flag fully clustered nodes,so that their evalution is skipped entirely.
 
     */
     auto continuationCriterion=[particle, percolationLength, centers, sizes](TreeNodeIndex idx)
     {     
          // Compute distance between particle and node nodeIdx
          // It might be beneficial to test whether a node is fully conatined in the ball around the particle.
          // If so, we can skip the distance calculation and add all particles in the node to the cluster.

          // Distance calculation based on node center and size.
          // If particle is in the node, the distance is zero.
          auto dX = minDistance(particle, centers[idx], sizes[idx]);
          double distance = 0.0;

          distance += dX[0]*dX[0];
          if (distance > percolationLength) return false;

          distance += dX[1]*dX[1];
          if (distance > percolationLength) return false;

          distance += dX[2]*dX[2];
          //Return true if distance below linking length
          return (distance < percolationLength);   
     };

     // Once a leaf node is reached, a brute force neighbour search is performed over all particles within the leaf.
     auto endpointAction=[i, particle, x, y, z, internalToLeaf, particleOffsets, percolationLength, clusterIdx, &particleClusterId, &fofQueue, startIndex, endIndex, leafCounts, &numNeighbors](TreeNodeIndex idx)
     {     
          Vec3<Tc> compParticle;
          Vec3<Tc> distances;
          Tc distance;
          auto leafIndex = internalToLeaf[idx];
          auto leafStart = particleOffsets[leafIndex];
          auto leafEnd = particleOffsets[leafIndex+1];

          // Check whether local particle is trying to access a non-local leaf. -> Don't want this!
          // If this happens, haloSearchExt_ is too small.
          if ((i>=startIndex) && (i<endIndex) && (leafCounts[leafIndex] > 0) && (leafStart==leafEnd))
          {
               std::cout << "Trying to access leaf with particles from local Particle." << std::endl;
          }
          
          for (auto l = leafStart; l<leafEnd; l++)
          {
               if (particleClusterId[l]) continue;
               compParticle[0] = x[l];
               compParticle[1] = y[l];
               compParticle[2] = z[l];
               distances = particle - compParticle;
               distance =
                    distances[0]*distances[0] + 
                    distances[1]*distances[1] + 
                    distances[2]*distances[2];
               if (distance < percolationLength)
               {
               particleClusterId[l] = clusterIdx;
               numNeighbors++;
               fofQueue.push(l);
               }
          }        
       
     };
     singleTraversal(childOffsets, continuationCriterion, endpointAction);
     return numNeighbors;
}

template<class ClusterIdType>
int removeSmallClusters(
     ClusterIdType* globalClusterIdx,
     std::vector<int>& localClusterMembers,
     std::vector<int>& globalClusterMembers,
     std::vector<ClusterIdType>& clusterMap,
     int memberThreshold,
     int startIndex,
     int endIndex
     )
{
     // Count members per cluster
	for (int pi=startIndex; pi < endIndex; ++pi) {
          localClusterMembers[globalClusterIdx[pi]-1] += 1;
          }

     // Allreduce members per cluster
     MPI_Allreduce(localClusterMembers.data(), globalClusterMembers.data(), localClusterMembers.size(), MPI_INT, MPI_SUM, MPI_COMM_WORLD);
     
     // Evaluate which clusters are too small
	for (int i=0; i<globalClusterMembers.size(); ++i) {
		if (globalClusterMembers[i] < memberThreshold) {
               globalClusterMembers[i] = 0;
          }
     }

	// Create a remapping
	clusterMap[0] = 0;
	ClusterIdType nClustersNew = 1;
	for (int i=0; i<globalClusterMembers.size(); ++i) {
		clusterMap[i] = nClustersNew;
		if (globalClusterMembers[i] == 0) {
			clusterMap[i] = 0;
		     }
		else {
			++nClustersNew;
		     }
	}

	// Remap the clusters
	for (int pi=startIndex; pi<endIndex; ++pi) {
		globalClusterIdx[pi] = clusterMap[globalClusterIdx[pi]-1];
	};

	return(nClustersNew-1);
	}

template<class ClusterIdType, class ClusterKeyType, class KeyType, class T, class Accelerator= CpuTag>
void fofMerge(
     std::vector<ClusterIdType>& localClusterIdx,
     ClusterIdType* globalClusterIdx,
     std::vector<ClusterKeyType>& localClusterKeys,
     Domain<KeyType, T, Accelerator>& domain,
     int myRank,
     int numRanks
)
{
     unionFindGlobal(
          localClusterIdx,
          localClusterKeys,
          domain,
          myRank
     );

     auto allKeys = assignCompactGlobalClusterIdx(
          localClusterKeys,
          myRank,
          numRanks,
          domain
     );
     
     // Remap cluster keys to global cluster IDs which start from 1 and increase sequentially
     std::set<ClusterKeyType> allClusterKeysSet;
     for (auto key : allKeys) {
          allClusterKeysSet.insert(key);
     }     
     std::vector<ClusterKeyType> uniqueGlobalKeys(allClusterKeysSet.begin(), allClusterKeysSet.end());     
     std::map<ClusterKeyType, ClusterIdType> keyToGlobalId;
     int idx = 1;
     for (const auto& key : uniqueGlobalKeys) {
          keyToGlobalId[key] = idx++;
     }

     // Assign global cluster IDs to local particles
     std::fill(globalClusterIdx, globalClusterIdx+domain.nParticlesWithHalos(), 0);
     for (int i = 0; i < domain.nParticles(); ++i) {
          globalClusterIdx[domain.startIndex()+i] = keyToGlobalId[localClusterKeys[domain.startIndex()+i]];
     }

     // Remove small clusters
     std::vector<int> localClusterMembers(uniqueGlobalKeys.size(), 0);
     std::vector<int> globalClusterMembers(uniqueGlobalKeys.size(), 0);
     std::vector<ClusterIdType> clusterMap(uniqueGlobalKeys.size(), 0);

     auto nClusters = removeSmallClusters(
          globalClusterIdx,
          localClusterMembers,
          globalClusterMembers,
          clusterMap,
          32,
          domain.startIndex(),
          domain.endIndex()
     );

     ClusterIdType maxClusterId = 0;
     for (int i = domain.startIndex(); i < domain.endIndex(); ++i) {
          if (globalClusterIdx[i] > maxClusterId) {
               maxClusterId = globalClusterIdx[i];
          }
     }
     ClusterIdType nClustersGlobal = 0;
     MPI_Reduce(&maxClusterId, &nClustersGlobal, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
     if (myRank == 0) {
          std::cout << "Found " << nClustersGlobal << " clusters." << std::endl;
     }     
}


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
template<class ClusterIdType, class KeyType, class T, class Tc, class Accelerator=CpuTag>
void fof(
     Domain<KeyType, T, Accelerator>& domain,
     ClusterIdType* globalClusterIdx,
     const Tc percolationLength,
     const Tc* x,
     const Tc* y,
     const Tc* z,
     const int myRank,
     const int numRanks
)
{
    auto nParticlesLocalWithHalos = domain.nParticlesWithHalos();
    std::vector<ClusterIdType> particleClusterIdx(nParticlesLocalWithHalos, 0);
    std::queue<int> fofQueue;
    int currentSeed = 0;
    ClusterIdType clusterIdx = 0;
    std::vector<unsigned> numNeighbors(domain.nParticles(), 0);
    
    for (int particleIdx = domain.startIndex(); particleIdx < domain.endIndex(); particleIdx++) {
        if (particleClusterIdx[particleIdx]) continue;
        clusterIdx += 1;
        particleClusterIdx[particleIdx] = clusterIdx;
        fofQueue.push(particleIdx);
        while (!(fofQueue.empty())) {
            currentSeed = fofQueue.front();
            fofQueue.pop();
            numNeighbors[particleIdx-domain.startIndex()] += particleFOF(
                domain,
                currentSeed,
                clusterIdx,
                x,
                y,
                z,
                percolationLength*percolationLength,
                fofQueue,
                particleClusterIdx,
                myRank
            );
            };
        };
     
     std::cout << "[" << myRank << "] " << *std::max_element(numNeighbors.begin(), numNeighbors.end()) << std::endl;
     std::vector<ClusterKeyType> localClusterKeys(nParticlesLocalWithHalos,0);

     fofMerge(
         particleClusterIdx,
         globalClusterIdx,
         localClusterKeys,
         domain,
         myRank,
         numRanks
     );
}

/*! @brief findNeighbors of particle number @p id within radius
 *
 * @tparam     T               coordinate type, float or double
 * @tparam     KeyType         32- or 64-bit Morton or Hilbert key type
 * @param[in]  i               the index of the particle for which to look for neighbors
 * @param[in]  x               particle x-coordinates in SFC order (as indexed by @p tree.layout)
 * @param[in]  y               particle y-coordinates in SFC order
 * @param[in]  z               particle z-coordinates in SFC order
 * @param[in]  tree            octree connectivity and particle indexing
 * @param[in]  box             coordinate bounding box that was used to calculate the Morton codes
 * @param[in]  percolationLength   linking length for the FoF algorithm
 * @return                     neighbor count of particle @p i, does not include self-reference; min return val is 0.
 */
template<class Tc, class Th, class KeyType>
HOST_DEVICE_FUN unsigned countNeighborsFOF(LocalIndex i,
                                           const Tc* x,
                                           const Tc* y,
                                           const Tc* z,
                                           const OctreeNsView<Tc, KeyType>& tree,
                                           const Box<Tc>& box,
                                           const Tc percolationLength)
{
    auto xi = x[i];
    auto yi = y[i];
    auto zi = z[i];

    auto radiusSq     = percolationLength * percolationLength;
    
    Vec3<Tc> particle{xi, yi, zi};
    unsigned numNeighbors = 0;

    auto pbc    = BoundaryType::periodic;
    bool anyPbc = box.boundaryX() == pbc || box.boundaryY() == pbc || box.boundaryZ() == pbc;
    bool usePbc = anyPbc && !insideBox(particle, {percolationLength,percolationLength,percolationLength}, box);

    auto overlapsPbc = [particle, radiusSq, centers = tree.centers, sizes = tree.sizes, &box](TreeNodeIndex idx)
    { return norm2(minDistance(particle, centers[idx], sizes[idx], box)) < radiusSq; };

    auto overlaps = [particle, radiusSq, centers = tree.centers, sizes = tree.sizes](TreeNodeIndex idx)
    { return norm2(minDistance(particle, centers[idx], sizes[idx])) < radiusSq; };

    auto searchBoxPbc =
        [i, particle, radiusSq, &tree, x, y, z, &numNeighbors, &box](TreeNodeIndex idx)
    {
        TreeNodeIndex leafIdx    = tree.internalToLeaf[idx];
        LocalIndex firstParticle = tree.layout[leafIdx];
        LocalIndex lastParticle  = tree.layout[leafIdx + 1];

        for (LocalIndex j = firstParticle; j < lastParticle; ++j)
        {
            if (j == i) { continue; }
            if (distanceSq<true>(x[j], y[j], z[j], particle[0], particle[1], particle[2], box) < radiusSq)
            {
                numNeighbors++;
            }
        }
    };

    auto searchBox = [i, particle, radiusSq, &tree, x, y, z, &numNeighbors, &box](TreeNodeIndex idx)
    {
        TreeNodeIndex leafIdx    = tree.internalToLeaf[idx];
        LocalIndex firstParticle = tree.layout[leafIdx];
        LocalIndex lastParticle  = tree.layout[leafIdx + 1];

        for (LocalIndex j = firstParticle; j < lastParticle; ++j)
        {
            if (j == i) { continue; }
            if (distanceSq<false>(x[j], y[j], z[j], particle[0], particle[1], particle[2], box) < radiusSq)
            {
                numNeighbors++;
            }
        }
    };

    if (usePbc) { singleTraversal(tree.childOffsets, overlapsPbc, searchBoxPbc); }
    else { singleTraversal(tree.childOffsets, overlaps, searchBox); }

    return numNeighbors;
}