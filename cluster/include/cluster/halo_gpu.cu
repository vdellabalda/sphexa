/*!
*  @brief Implements GPU versions of cluster properties computations.
*/
 
#include "cstone/cuda/annotation.hpp"
#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/primitives/primitives_gpu.h"

#include "halo_gpu.h"
#include "definitions.h"

namespace halo
{

// Overload for double (your custom implementation)
__device__ double atomicAddCustom(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    
    return __longlong_as_double(old);
}

// Overload for float (delegates to CUDA's implementation)
__device__ float atomicAddCustom(float* address, float val)
{
    return atomicAdd(address, val);
}

/*
template<class ClusterData, class HaloData, class DomainType>
void reassignHalosGPU(
    ClusterData& c,
    HaloData&    h,
    const DomainType& domain,
    int numRanks
)
{
    auto nParticles = c.getNumParticles();
    memcpyD2D(rawPtr(c.devData.halo_id)+domain.startIndex(), nParticles, rawPtr(c.devData.idBuf));
    cstone::sortGpu(rawPtr(c.devData.idBuf), rawPtr(c.devData.idBuf)+nParticles);

    auto numUniqueIds = cstone::uniqueCountGpu(rawPtr(c.devData.idBuf), rawPtr(c.devData.idBuf)+nParticles);
    cstone::runLengthEncodeGpu(numUniqueIds, rawPtr(c.devData.idBuf), rawPtr(h.devData.globalId), rawPtr(c.devData.sizes), rawPtr(c.devData.numClusters));
    cstone::sequenceGpu(rawPtr(h.devData.localId), numUniqueIds, unsigned(0));

    int localCount = numUniqueIds-1; // exclude halo ID 0
    std::vector<int> recvCounts(numRanks);
    std::vector<int> displs(numRanks), counts(numRanks);
    std::fill(counts.begin(), counts.end(), 1);
    std::iota(displs.begin(), displs.end(), 0);

    mpiAllgatherv(&localCount, 1, recvCounts.data(), counts.data(), displs.data(), MPI_COMM_WORLD);

    displs[0] = 0;
    for (int i = 1; i < numRanks; ++i)
        displs[i] = displs[i - 1] + recvCounts[i - 1];
    int totalCount = displs.back() + recvCounts.back();
        
    // Allgather keys
    std::vector<ClusterKeyType> globalKeysHost(totalCount);
    std::vector<ClusterIdType> globalKeyCountsHost(totalCount);
    std::vector<ClusterKeyType> localKeysHost(localCount);
    std::vector<ClusterIdType> localKeyCountsHost(localCount);
    memcpyD2H(rawPtr(c.devData.uniqueKeys)+1, localCount, localKeysHost.data());
    
    mpiAllgatherv(
        localKeysHost.data(), localCount,
        globalKeysHost.data(), recvCounts.data(), displs.data(),
        MPI_COMM_WORLD
    );
                
    memcpyH2D(globalKeysHost.data(), totalCount, rawPtr(c.devData.globalClusterKeys));

    memcpyH2D(h.keyOwnerPair.data(), totalCount, rawPtr(h.devData.keyOwnerPair));
    
    // Allgather key counts
    memcpyD2H(rawPtr(c.devData.keyCounts)+1, localCount, localKeyCountsHost.data());
    mpiAllgatherv(
        localKeyCountsHost.data(), localCount,
        globalKeyCountsHost.data(), recvCounts.data(), displs.data(),
        MPI_COMM_WORLD
    );
    memcpyH2D(globalKeyCountsHost.data(), totalCount, rawPtr(c.devData.idBuf));

    h.keyOwnerPair.resize(totalCount);
    h.devData.keyOwnerPair.resize(totalCount);
    for (int i = 0; i < numRanks; ++i)
    {
        for (int j = 0; j < recvCounts[i]; ++j)
        {
            h.keyOwnerPair[displs[i] + j] = {globalKeyCounts[displs[i]+j], static_cast<unsigned>(i)};
        }
    }
    memcpyH2D(rawPtr(h.keyOwnerPair), totalCount, rawPtr(h.devData.keyOwnerPair));

    // Reduce Cluster Keys such that counts are summed for identical keys and ownership is assigned
    // based on which rank has the largest count for that key
    // Values are (counts, ownerRank), keys are cluster keys
    // Output written to h.devData.ownership for clusteres which are represented locally
    //std::pair<ClusterKeyType*, ClusterIdType*> new_iterators = cstone::findHaloOwnershipGpu(
    //    rawPtr(c.devData.globalClusterKeys), rawPtr(c.devData.globalClusterKeys)+totalCount,
    //    rawPtr(c.devData.keyOwnerPair),
    //    rawPtr(c.devData.uniqueKeys),
    //    rawPtr(c.devData.ownership)
    //);

    // Reduce Cluster Keys such that counts are summed for identical keys and ownership is assigned
    // based on which rank has the largest count for that key
    // Values are (counts, ownerRank), keys are cluster keys
    // Output written to h.devData.ownership for clusteres which are represented locally
    

    
}
    */

template<class Tc, class Tm, class T>
HOST_DEVICE_FUN void centerOfMass(
    unsigned idx,
    const Tc* x,
    const Tc* y,
    const Tc* z,
    const Tm* mass,
    const T* haloIds,
    Tc* haloCenterX,
    Tc* haloCenterY,
    Tc* haloCenterZ,
    Tm* haloMass)
{
    unsigned haloId = haloIds[idx];
    if (haloId == 0) return;
    Tm m = mass[idx];
    atomicAdd(&haloCenterX[haloId], x[idx] * m);
    atomicAdd(&haloCenterY[haloId], y[idx] * m);
    atomicAdd(&haloCenterZ[haloId], z[idx] * m);
    atomicAdd(&haloMass[haloId], m);
};

template<class Tc, class Tm, class T>
HOST_DEVICE_FUN void haloVelocity(
    unsigned idx,
    const Tc* vx,
    const Tc* vy,
    const Tc* vz,
    const Tm* mass,
    const T* haloIds,
    Tc* haloVelocityX,
    Tc* haloVelocityY,
    Tc* haloVelocityZ)
{
    unsigned haloId = haloIds[idx];
    if (haloId == 0) return;
    Tm m = mass[idx];
    atomicAdd(&haloVelocityX[haloId], vx[idx] * m);
    atomicAdd(&haloVelocityY[haloId], vy[idx] * m);
    atomicAdd(&haloVelocityZ[haloId], vz[idx] * m);
};

template<class Tc, class Tv, class Tm, class T>
__global__ void haloPropertiesKernel(
    size_t first,
    size_t last,
    const Tc* x,
    const Tc* y,
    const Tc* z,
    const Tv* vx,
    const Tv* vy,
    const Tv* vz,
    const Tm* mass,
    const T* haloIds,
    Tm* haloCenterX,
    Tm* haloCenterY,
    Tm* haloCenterZ,
    Tv* haloVelocityX,
    Tv* haloVelocityY,
    Tv* haloVelocityZ,
    Tm* haloMass
    )
{   
    unsigned idx = first + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= last) return;

    unsigned haloId = haloIds[idx];
    if (haloId == 0) return;
    Tm m = mass[idx];

    atomicAddCustom(&haloCenterX[haloId], x[idx] * m);
    atomicAddCustom(&haloCenterY[haloId], y[idx] * m);
    atomicAddCustom(&haloCenterZ[haloId], z[idx] * m);
    atomicAddCustom(&haloMass[haloId], m);
    atomicAddCustom(&haloVelocityX[haloId], vx[idx] * m);
    atomicAddCustom(&haloVelocityY[haloId], vy[idx] * m);
    atomicAddCustom(&haloVelocityZ[haloId], vz[idx] * m);
};

#define COMPUTE_HALO_PROPS(Tc, Tv, Tm, T)                                                                            \
    template __global__ void haloPropertiesKernel(size_t first, size_t last, const Tc* x, const Tc* y, const Tc* z,  \
                                       const Tv* vx, const Tv* vy, const Tv* vz, const Tm* mass, const T* haloIds,   \
                                       Tm* haloCenterX, Tm* haloCenterY, Tm* haloCenterZ,                            \
                                       Tv* haloVelocityX, Tv* haloVelocityY, Tv* haloVelocityZ, Tm* haloMass)

COMPUTE_HALO_PROPS(double, double, double, unsigned);
COMPUTE_HALO_PROPS(double, double, float, unsigned);
COMPUTE_HALO_PROPS(double, float, double, unsigned);
COMPUTE_HALO_PROPS(float, double, double, unsigned);
COMPUTE_HALO_PROPS(double, float, float, unsigned);
COMPUTE_HALO_PROPS(float, double, float, unsigned);
COMPUTE_HALO_PROPS(float, float, double, unsigned);
COMPUTE_HALO_PROPS(float, float, float, unsigned);

template<class ParticleDataset, class ClusterDataset, class HaloDataset>
void haloPropertiesGPU(
    size_t first,
    size_t last,
    ParticleDataset& d,
    ClusterDataset& c,
    HaloDataset& h
)
{
    unsigned numThreads = 256;
    unsigned numBlocks = (last - first + numThreads - 1) / numThreads;
    haloPropertiesKernel<<<numBlocks, numThreads>>>(
        first,
        last,
        rawPtr(d.devData.x),
        rawPtr(d.devData.y),
        rawPtr(d.devData.z),
        rawPtr(d.devData.vx),
        rawPtr(d.devData.vy),
        rawPtr(d.devData.vz),
        rawPtr(d.devData.m),
        rawPtr(c.devData.halo_id),
        rawPtr(h.devData.centerX),
        rawPtr(h.devData.centerY),
        rawPtr(h.devData.centerZ),
        rawPtr(h.devData.velocityX),
        rawPtr(h.devData.velocityY),
        rawPtr(h.devData.velocityZ),
        rawPtr(h.devData.mass)
    );
}
template void haloPropertiesGPU(
    size_t first,
    size_t last,
    sphexa::ParticlesData<cstone::GpuTag>& d,
    cluster::ClusterData<cstone::GpuTag>& c,
    halo::HaloData<cstone::GpuTag>& h
);
} // namespace halo