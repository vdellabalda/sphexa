/*! @file
 * @brief Cluster key and edge definitions
 *
 * This file defines the types and functions used to create cluster keys and edges.
 * A cluster key is a unique identifier for a cluster, while a cluster edge represents
 * a connection between two clusters.
 * 
 * @author Vincente Della Balda
 */

//#include <cstdint>
//#include <tuple>
//#include <vector>
//#include <algorithm>
//#include <map>
//#include <set>
//#include <mpi.h>

#include "cstone/domain/domain.hpp"
#include "definitions.h"

namespace cluster
{

/*! @brief Create a cluster key from rank and local cluster ID
 *
 * @param rank            MPI rank
 * @param localClusterID  Local cluster ID
 * @return                Cluster key
 *
 * The cluster key is created by shifting the rank to the left by 32 bits and OR'ing it with the local cluster ID.
 */ 
HOST_DEVICE_FUN inline ClusterKeyType makeClusterKey(
    int rank,
    ClusterIdType localClusterID
)
{
    return (static_cast<ClusterKeyType>(rank) << 32) | static_cast<ClusterKeyType>(localClusterID);
}

/*! @brief Extract rank from cluster key
 *
 * @param clusterKey  Cluster key
 * @return            MPI Rank
 *
 * The rank is extracted by shifting the cluster key to the right by 32 bits.
 */
HOST_DEVICE_FUN inline int getRankFromClusterKey(
    ClusterKeyType clusterKey
)
{
    return static_cast<int>(clusterKey >> 32);
}

/*! @brief Extract local cluster ID from cluster key
 *
 * @param clusterKey  Cluster key
 * @return            Local cluster ID
 *
 * The local cluster ID is extracted by masking the lower 32 bits of the cluster key.
 */
HOST_DEVICE_FUN inline ClusterIdType getLocalClusterIdFromClusterKey(
    ClusterKeyType clusterKey
)
{
    return static_cast<ClusterIdType>(clusterKey & 0xFFFFFFFF);
}
} // namespace cluster