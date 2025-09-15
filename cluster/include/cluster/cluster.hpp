/*! @file
 * @brief Cluster key and edge definitions
 *
 * This file defines the types and functions used to create cluster keys and edges.
 * A cluster key is a unique identifier for a cluster, while a cluster edge represents
 * a connection between two clusters.
 * 
 * @author Vincente Della Balda
 */

#include <cstdint>
#include <tuple>
#include <vector>
#include <algorithm>
#include <map>
#include <set>
#include <mpi.h>

#include "cstone/domain/domain.hpp"

using ClusterKeyType = uint64_t;
using ClusterIdType = uint32_t;

using namespace cstone;

typedef struct ClusterEdge
{
    ClusterKeyType localCluster;
    ClusterKeyType remoteCluster;
    bool operator<(const ClusterEdge& other) const {
        return std::tie(localCluster, remoteCluster) < std::tie(other.localCluster, other.remoteCluster);
    };
} clusterEdge;

/*! @brief Create a cluster key from rank and local cluster ID
 *
 * @param rank            MPI rank
 * @param localClusterID  Local cluster ID
 * @return                Cluster key
 *
 * The cluster key is created by shifting the rank to the left by 32 bits and OR'ing it with the local cluster ID.
 */ 
template<class ClusterIdType>
ClusterKeyType makeClusterKey(
    int rank,
    ClusterIdType localClusterID
)
{
    return (static_cast<ClusterKeyType>(rank) << 32) | static_cast<ClusterKeyType>(localClusterID);
}

/*! @brief Extract rank from cluster key
 *
 * @param clusterKey  Cluster key
 * @return           Rank
 *
 * The rank is extracted by shifting the cluster key to the right by 32 bits.
 */
template<class ClusterKeyType>
int getRankFromClusterKey(
    ClusterKeyType clusterKey
)
{
    return static_cast<int>(clusterKey >> 32);
}

/*! @brief Extract local cluster ID from cluster key
 *
 * @param clusterKey  Cluster key
 * @return           Local cluster ID
 *
 * The local cluster ID is extracted by masking the lower 32 bits of the cluster key.
 */
template<class ClusterKeyType>
ClusterIdType getLocalClusterIdFromClusterKey(
    ClusterKeyType clusterKey
)
{
    return static_cast<ClusterIdType>(clusterKey & 0xFFFFFFFF);
}

/*! @brief communicate halo cluster keys across ranks
 *
 * @param localClusterKeys     local cluster keys
 * @param globalClusterKeys    global cluster keys
 * @param domain               domain object for halo exchange
 * 
 * This function gathers the halo cluster keys from all ranks and communicates them to the other ranks.
 */
template<class KeyType, class T, class Accelerator = CpuTag, class ClusterKeyType>
void gatherHaloClusterKeys(
     const std::vector<ClusterKeyType>& localClusterKeys, 
     std::vector<ClusterKeyType>& globalClusterKeys,
     Domain<KeyType, T, Accelerator>& domain) 
{
     const int nLocal = domain.nParticles();
     const int nHalo = domain.nParticlesWithHalos() - nLocal;
     assert(static_cast<int>(localClusterKeys.size()) == domain.nParticlesWithHalos());

     const int nHaloStart = domain.startIndex();
     const int nHaloEnd = nHalo - nHaloStart;
     
     std::vector<ClusterKeyType> sendBuffer(domain.nParticlesWithHalos());
     std::vector<ClusterKeyType> recvBuffer(domain.nParticlesWithHalos());
     std::fill(globalClusterKeys.begin(), globalClusterKeys.end(), -1);

     // Fill send buffer: local particles keep their cluster keys, halos set to dummy
     std::copy(localClusterKeys.begin()+nHaloStart, localClusterKeys.end()-nHaloEnd, globalClusterKeys.begin()+nHaloStart);

     // Perform the halo exchange
     domain.exchangeHalos(std::tie(globalClusterKeys), sendBuffer, recvBuffer);
}

/*! @brief Find the root of a cluster using path compression
*
* @param clusterParents  Vector of cluster parents
* @param clusterKey      Cluster key to find the root for
* @return                Root of the cluster
*
* This function implements path compression to optimize the union-find algorithm.
*/
template<class ClusterKeyType>
ClusterKeyType findRoot(
    std::vector<int>& clusterParents,
    ClusterKeyType clusterKey
)
{   //std::cout << "Parent of " << clusterKey << " is " << clusterParents[clusterKey] << std::endl;
    if (clusterParents[clusterKey] != clusterKey) {
        clusterParents[clusterKey] = findRoot(clusterParents, clusterParents[clusterKey]);
    }
    return clusterParents[clusterKey];
}

/*! @brief Union-Find algorithm to merge clusters
 *
 * @param clusterKeys     Vector of cluster keys
 * @param clusterEdges    Vector of cluster edges
 * @param clusterParents  Vector of cluster parents
 * @param clusterSizes    Vector of cluster sizes
 *
 * This function implements the union-find algorithm to merge clusters based on the provided edges.
 */
template<class ClusterKeyType>
void unionFind(
    std::vector<clusterEdge>& clusterEdges,
    std::vector<int>& clusterParents,
    std::vector<int>& clusterSizes,
    std::map<ClusterKeyType, int>& keyToIndex,
    std::vector<ClusterKeyType>& indexToKey,
    const int myRank
)
{
    //Iterate through the edges and merge clusters
    for (auto edge : clusterEdges) {
        ClusterKeyType root1 = findRoot(clusterParents, edge.localCluster);
        ClusterKeyType root2 = findRoot(clusterParents, edge.remoteCluster);

        if (root1 != root2) {
            // Merge clusters not according to size but prioritize the smaller rank, i.e. lower cluster key.
            if (indexToKey[root1] > indexToKey[root2]) {
                std::swap(root1, root2);
            }
            clusterParents[root2] = root1;
            clusterSizes[root1] += clusterSizes[root2];
        }
    }
    return;
}

template<class ClusterKeyType, class ClusterIdType, class KeyType, class T, class Accelerator= CpuTag>
void unionFindGlobal(
    std::vector<ClusterIdType>& localClusterIdx,
    std::vector<ClusterKeyType>& localClusterKeys,
    Domain<KeyType, T, Accelerator>& domain,
    int myRank
)
{
    std::vector<ClusterKeyType> globalClusterKeys(localClusterIdx.size());

    std::set<ClusterKeyType> allClusterKeys;
    std::map<ClusterKeyType, int> keyToIndex;
    std::vector<ClusterKeyType> indexToKey;

    std::set<clusterEdge> clusterEdges;     
    ClusterKeyType localClusterKey;
    ClusterKeyType globalClusterKey;
    clusterEdge currentEdge;

    std::vector<clusterEdge> clusterEdgesVec;
    std::vector<int> clusterParents(domain.nParticles());
    std::vector<int> clusterSizes(domain.nParticles());

    for (int i = 0; i < clusterParents.size(); ++i) {
        clusterParents[i] = i;
        clusterSizes[i] = 1;
    };

    bool clusterChange = true;
    ClusterKeyType updatedClusterKey;
    
    for (int i=0; i<domain.nParticlesWithHalos(); ++i) {
        localClusterKeys[i] = makeClusterKey(myRank, localClusterIdx[i]);
    }

    int idx = 0;
    int union_findCount = 1;
    // Cluster Key Propagation until convergence
    while (clusterChange) {
        clusterChange = false;
        clusterEdges.clear();
        clusterEdgesVec.clear();
        
        // Exchange Cluster Keys of Halo Particles
        gatherHaloClusterKeys(
            localClusterKeys,
            globalClusterKeys,
            domain
        );

        // Create mapping from local-halo Cluster Keys to local Cluster IDs
        // 1. Find unique Cluster Keys
        for (int i=0; i<domain.nParticlesWithHalos(); ++i) {
            if (!(localClusterIdx[i])) continue;
            allClusterKeys.insert(globalClusterKeys[i]);
        }
        // 2. Create map
        for (const auto& key : allClusterKeys) {
            if (keyToIndex.find(key) == keyToIndex.end()) {
                if (union_findCount>1) std::cout << "[" << myRank << "] Adding Cluster Key " << key << " to indexToKey" << std::endl;
                indexToKey.push_back(key);
                keyToIndex[key] = idx++;
            }
        }
        
        // Find edges according to Cluster assignment of Halo Particles
        for (int i=0; i<domain.startIndex(); ++i) {
            if (!(localClusterIdx[i])) continue;
            localClusterKey = localClusterKeys[i];
            globalClusterKey = globalClusterKeys[i];
            if (globalClusterKey < localClusterKey) std::swap(localClusterKey, globalClusterKey);
            currentEdge.localCluster = keyToIndex[localClusterKey];
            currentEdge.remoteCluster = keyToIndex[globalClusterKey];
            clusterEdges.insert(currentEdge);
        }

        for (int i=domain.endIndex(); i<domain.nParticlesWithHalos(); ++i) {
            if (!(localClusterIdx[i])) continue;
            localClusterKey = localClusterKeys[i];
            globalClusterKey = globalClusterKeys[i];
            if (globalClusterKey < localClusterKey) std::swap(localClusterKey, globalClusterKey);
            currentEdge.localCluster = keyToIndex[localClusterKey];
            currentEdge.remoteCluster = keyToIndex[globalClusterKey];
            clusterEdges.insert(currentEdge);
        }

        // Find local union of disjoint sets
        clusterEdgesVec.resize(clusterEdges.size());
        std::copy(clusterEdges.begin(), clusterEdges.end(), clusterEdgesVec.begin());

        unionFind(
            clusterEdgesVec,
            clusterParents,
            clusterSizes,
            keyToIndex,
            indexToKey,
            myRank
        );
        
        // Update Cluster Key of local particles
        for (int i = domain.startIndex(); i < domain.endIndex(); ++i) {
            updatedClusterKey = indexToKey[findRoot(clusterParents, keyToIndex[localClusterKeys[i]])];
            if (localClusterKeys[i] != updatedClusterKey) {
                localClusterKeys[i] = updatedClusterKey;
                clusterChange = true;            
            }
        }

        // Communicate whether any Cluster Key has been changed
        bool globalChange;
        MPI_Allreduce(&clusterChange, &globalChange, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
        clusterChange = globalChange;
        //if (myRank==0) std::cout << "Union-Find Iteration " << union_findCount++ << std::endl;
    }
}

template<class ClusterKeyType, class KeyType, class T, class Accelerator=CpuTag>
std::vector<ClusterKeyType> assignCompactGlobalClusterIdx(
    const std::vector<ClusterKeyType>& localClusterKeys,
    int myRank,
    int numRanks,
    Domain<KeyType, T, Accelerator>& domain
    ) 
    {

    // Create unique keys from local particles
    std::set<ClusterKeyType> clusterKeysSet;
    for (int i = domain.startIndex(); i < domain.endIndex(); ++i) {
        clusterKeysSet.insert(localClusterKeys[i]);
    }

    std::vector<ClusterKeyType> uniqueLocalKeys(clusterKeysSet.begin(), clusterKeysSet.end());
    std::sort(uniqueLocalKeys.begin(), uniqueLocalKeys.end());

    // Allgather Key counts
    int localCount = static_cast<int>(uniqueLocalKeys.size());
    std::vector<int> recvCounts(numRanks), displs(numRanks);
    MPI_Allgather(&localCount, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    displs[0] = 0;
    for (int i = 1; i < numRanks; ++i)
        displs[i] = displs[i - 1] + recvCounts[i - 1];
    int totalCount = displs.back() + recvCounts.back();
    //if (myRank==0) std::cout << "Global unique key count: " << totalCount << std::endl;

    // Allgather keys
    std::vector<ClusterKeyType> allKeysFlat(totalCount);
    MPI_Allgatherv(uniqueLocalKeys.data(), localCount, MPI_UINT64_T,
                   allKeysFlat.data(), recvCounts.data(), displs.data(), MPI_UINT64_T,
                   MPI_COMM_WORLD);

    return allKeysFlat;
}