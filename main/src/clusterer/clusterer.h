/*! @file
 * @brief Clusterer initialization
 *
 * @author Vincente Della Balda <vinc.dellabalda@gmail.com>
 */

#pragma once

#include <memory>

#include "iclusterer.hpp"
#include "init/settings.hpp"
#include "sphexa/simulation_data.hpp"

namespace sphexa
{

template<class DomainType, class ParticleDataType>
struct ClustLib
{

    using ClustPtr = std::unique_ptr<Clusterer<DomainType, ParticleDataType>>;

    static ClustPtr makeDarkClust(std::ostream& output, size_t rank, bool avClean);
    static ClustPtr makeDarkClustDens(std::ostream& output, size_t rank, bool avClean);
};

} // namespace sphexa