#include "definitions.h"
#include "cluster_utils.hpp"

namespace cluster
{

template<typename Tin>
size_t uniquify(Tin* input, size_t n)
{
    std::sort(input, input + n);
    auto newEnd = std::unique(input, input + n);
    return std::distance(input, newEnd);
}
template size_t uniquify(ClusterKeyType* input, size_t n);
template size_t uniquify(EdgeType* input, size_t n);


template<typename Tin, typename Tout>
size_t runLengthEncode(Tin* input, size_t n, Tin* unique, Tout* counts)
{
    std::sort(input, input + n);
    size_t uniqueCount = 0;
    for (size_t i = 0; i < n; ++i)
    {
        size_t count = 1;
        while (i < n && input[i] == input[i+1])
        {
            count++;
            i++;
        }
        unique[uniqueCount] = input[i];
        counts[uniqueCount] = count;
        uniqueCount++;
    }
    return uniqueCount;
}
template size_t runLengthEncode(ClusterKeyType* input, size_t n, ClusterKeyType* unique, ClusterIdType* counts);

void assignClusterKey(
    ClusterIdType* clusterIdx,
    ClusterKeyType* clusterKeys,
    ClusterIdType* flagged,
    size_t n,
    int rank
)
{
    ClusterIdType clusterId;
    for (size_t i = 0; i < n; ++i)
    {
        clusterId = clusterIdx[i];
        if (clusterId != 0)
        {
            flagged[i] = 1; // mark as clustered
            clusterKeys[i] = makeClusterKey(rank, clusterId);
        }
    }
}

size_t allGathervSetup(
     size_t localCount,
     int* recvCounts,
     int* displs
     )
{
     int numRanks;
     MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
 
     MPI_Allgather(&localCount, 1, MPI_INT, recvCounts, 1, MPI_INT, MPI_COMM_WORLD); 
     displs[0] = 0;
     for (int i = 1; i < numRanks; ++i)
         displs[i] = displs[i - 1] + recvCounts[i - 1];
     int totalCount = displs[numRanks-1] + recvCounts[numRanks-1];
 
     return totalCount;
}
} // namespace cluster