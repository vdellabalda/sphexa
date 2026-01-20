#pragma once

#include <cassert>
#include <type_traits>
#include "cstone/cuda/annotation.hpp"
#include "cstone/util/array.hpp"

namespace cluster

{

/*! @brief
 * Controls the types, used throughout the cluster module.
 */

using ClusterKeyType = uint64_t; // for global unique cluster keys
using ClusterIdType = unsigned; // for local cluster IDs
using EdgeType = util::array<ClusterKeyType, 2>; // for cluster edges to perform union-find

}