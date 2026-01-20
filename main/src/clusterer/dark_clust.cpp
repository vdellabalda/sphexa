/*! @file
 * @brief Translation unit for the dark clusterer initializer
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 * @author Vincente Della Balda <vinc.dellabalda@gmail.com>
 */

#include "sph/types.hpp"
#include "clusterer.h"
#include "dark_clust.hpp"

namespace sphexa
{

template<class DomainType, class ParticleDataType>
std::unique_ptr<Clusterer<DomainType, ParticleDataType>>
ClustLib<DomainType, ParticleDataType>::makeDarkClust(std::ostream& output, size_t rank, bool avClean)
{
    return std::make_unique<darkClust<DomainType, ParticleDataType>>(output, rank, avClean);
}

#ifdef USE_CUDA
template struct ClustLib<cstone::Domain<SphTypes::KeyType, SphTypes::CoordinateType, cstone::GpuTag>,
                        SimulationData<cstone::GpuTag>>;
#else
template struct ClustLib<cstone::Domain<SphTypes::KeyType, SphTypes::CoordinateType, cstone::CpuTag>,
                        SimulationData<cstone::CpuTag>>;
#endif

} // namespace sphexa
