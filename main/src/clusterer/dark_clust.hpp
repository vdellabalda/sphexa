/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief A dark clusterer class
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 * @author Jose A. Escartin <ja.escartin@gmail.com>
 */

#pragma once

#include <variant>

#include "cstone/fields/field_get.hpp"
#include "cluster/cluster_data.hpp"
#include "sph/particles_data.hpp"
#include "sph/sph.hpp"

#include "iclusterer.hpp"

namespace sphexa
{

using namespace sph;
using util::FieldList;

template<class DomainType, class DataType>
class darkClust : public Clusterer<DomainType, DataType>
{
protected:
    using Base = Clusterer<DomainType, DataType>;
    using Base::timer;

    using T             = typename DataType::RealType;
    using KeyType       = typename DataType::KeyType;

    using Acc       = typename DataType::AcceleratorType;

    GroupData<Acc> groups_;

public:
    darkClust(std::ostream& output, size_t rank)
        : Base(output, rank)
    {
    }
    
    /*void findClusters_MPI(DomainType& domain, ParticleDataType& d, double percolationLength, int numRanks)
    {   
        timer.start();
        // compute the clusters
        fof(
            domain,
            d.hydro.halo_id.data(),
            percolationLength,
            d.hydro.x.data(),
            d.hydro.y.data(),
            d.hydro.z.data(),
            rank_,
            numRanks
        );

        timer.step("computeClusters");

        //out << "# Clustering: " << timer.sumOfSteps() << "s\n";
    }*/

    void findClusters(DomainType& domain, DataType& simData)
    {   
        timer.start();
        auto& d = simData.hydro;
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();
        findNeighborsSfc(first, last, d, domain.box());
        computeGroups(first, last, d, domain.box(), groups_);
        // compute the clusters
        computeClusterId(
            groups_.view(),
            simData.hydro,
            simData.clust,
            domain.box()
        );

        timer.step("computeClusters");

        //out << "# Clustering: " << timer.sumOfSteps() << "s\n";
    }

    void saveFields(IFileWriter* writer, size_t first, size_t last, DataType& simData,
                    const cstone::Box<T>& /*box*/) override
    {
        Base::outputAllocatedFields(writer, first, last, simData);
        timer.step("FileOutput");
    }
};

} // namespace sphexa