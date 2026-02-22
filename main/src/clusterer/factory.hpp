/*! @file
 * @brief Evaluate choice of clusterer
 *
 * @author Vincente Della Balda <vinc.dellabalda@gmail.com>
 */

#pragma once

#include "clusterer.h"

namespace sphexa
{

template<class DomainType, class ParticleDataType>
std::unique_ptr<Clusterer<DomainType, ParticleDataType>>
clustFactory(const std::string& choice, bool subCluster, std::ostream& output, size_t rank)
{
    if (choice == "dark") { return ClustLib<DomainType, ParticleDataType>::makeDarkClust(output, rank, subCluster); }

    throw std::runtime_error("Unknown clusterer choice: " + choice);
}

} // namespace sphexa
