#include "definitions.h"

namespace cluster
{

/*! @brief Find the root of a cluster using path compression
*
* @param clusterParents  Vector of cluster parents
* @param clusterKey      Cluster key to find the root for
* @return                Root of the cluster
*
* This function implements path compression to optimize the union-find algorithm.
*/
ClusterKeyType findRoot(
    ClusterIdType* clusterParents,
    ClusterKeyType clusterKey
)
{   
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
void unionFind(
    EdgeType* clusterEdges,
    size_t numEdges,
    ClusterIdType* clusterParents,
    ClusterIdType* clusterSizes,
    ClusterKeyType* indexToKey
)
{
    //Iterate through the edges and merge clusters
    for (size_t i = 0; i < numEdges; ++i) {
        ClusterKeyType root1 = findRoot(clusterParents, clusterEdges[i][0]);
        ClusterKeyType root2 = findRoot(clusterParents, clusterEdges[i][1]);

        if (root1 != root2) {
            if (indexToKey[root1] > indexToKey[root2]) {
                std::swap(root1, root2);
            }
            clusterParents[root2] = root1;
            clusterSizes[root1] += clusterSizes[root2];
        }
    }
    return;
}
} // namespace cluster