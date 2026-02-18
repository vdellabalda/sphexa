#pragma once

#include <array>
#include "mpi.h"

namespace cluster {

struct ClusterInfo {
    unsigned clusterId;
    uint64_t globalOffset;    // Starting position in the global file
    uint64_t globalCount;     // Total particles in this cluster across all ranks
    uint32_t localCount;      // Number of particles this rank contributes to this cluster
    uint32_t localOffset;     // Starting position in the local arrays for this cluster
    uint32_t haloOffset;      // Offset of the particles in the full halo_id array for this cluster
};

struct HDF5HostData {
    std::vector<double> x, y, z;
    std::vector<float> vx, vy, vz;
    std::vector<float> m;
    std::vector<unsigned> halo_id;
    std::vector<uint64_t> id;
};

inline uint32_t computeHaloOffset(uint32_t localCount) {
    // Compute the offset in the halo_id array for this cluster
    // This is needed because particles for different clusters are stored contiguously
    // in the halo_id array, and we need to know where the particles for this cluster start
    uint32_t haloOffset = 0;
    MPI_Exscan(&localCount, &haloOffset, 1, MPI_UINT32_T, MPI_SUM, MPI_COMM_WORLD);
    return haloOffset;
};
}