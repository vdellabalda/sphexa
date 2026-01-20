#include "sph/particles_data.hpp"
#include "sph/types.hpp"
#include "cluster_data.hpp"
#include "halo_data.hpp"

namespace halo
{   
    template<class ParticleDataset, class ClusterDataset, class HaloDataset>
    void haloPropertiesGPU(
        size_t first,
        size_t last,
        ParticleDataset& d,
        ClusterDataset& c,
        HaloDataset& h);
} // namespace halo