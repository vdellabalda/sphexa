#include "halo_gpu.h"

namespace halo
{
template<class ParticleDataset, class ClusterDataset, class HaloDataset>
void computeHaloProperties(
    size_t first,
    size_t last,
    ParticleDataset& d,
    ClusterDataset& c,
    HaloDataset& h)
{
    if constexpr (cstone::HaveGpu<typename ParticleDataset::AcceleratorType>{})
        { haloPropertiesGPU(first, last, d, c, h); }
    else { }
}
}