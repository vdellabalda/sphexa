/*! @file
 * @brief A dark clusterer class
 *
 * @author Vincente Della Balda <vinc.dellabalda@gmail.com>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <variant>

#include "cstone/fields/field_get.hpp"
#include "cstone/domain/domain.hpp"
#include "sph/particles_data.hpp"
#include "sph/find_neighbors.hpp"
#include "sph/groups.hpp"
#include "sph/positions.hpp"
#include "sph/ts_global.hpp"

#include "cluster/cluster.hpp"
#include "cluster/cluster_data.hpp"
#include "cluster/halo.hpp"
#include "cluster/halo_data.hpp"

#include "iclusterer.hpp"

namespace sphexa
{

using namespace sph;
using namespace cluster;
using namespace halo;
using util::FieldList;

template<class DomainType, class ParticleDataType>
class darkClust : public Clusterer<DomainType, ParticleDataType>
{
protected:
    using Base = Clusterer<DomainType, ParticleDataType>;
    using Base::timer;

    using T              = typename ParticleDataType::RealType;
    using KeyType        = typename ParticleDataType::KeyType;

    using Acc       = typename ParticleDataType::AcceleratorType;

    GroupData<Acc> groups_;

    using ConservedFields = FieldList<"halo_id">;

    using DependentFields =
        FieldList<"flagged", "idBuf", "keyBuf", "localClusterKeys", "globalClusterKeys">;


public:
    darkClust(std::ostream& output, size_t rank, bool avClean)
        : Base(output, rank)
    {
    }

    std::vector<std::string> conservedFields() const override
    {
        std::vector<std::string> ret{"halo_id"};
        for_each_tuple([&ret](auto f) { ret.push_back(f.value); }, make_tuple(ConservedFields{}));
        return ret;
    }

    void activateFields(ParticleDataType& simData) override
    {
        auto& c = simData.clust;

        //! @brief Fields accessed in domain sync are not part of extensible lists.
        c.setConserved("halo_id");
        c.setDependent("flagged", "idBuf", "keyBuf", "localClusterKeys", "globalClusterKeys");
        std::apply([&c](auto... f) { c.setConserved(f.value...); }, make_tuple(ConservedFields{}));
        std::apply([&c](auto... f) { c.setDependent(f.value...); }, make_tuple(DependentFields{}));


        c.devData.setConserved("halo_id");
        c.devData.setDependent("flagged", "idBuf", "keyBuf", "localClusterKeys", "globalClusterKeys");
        std::apply([&c](auto... f) { c.devData.setConserved(f.value...); }, make_tuple(ConservedFields{}));
        std::apply([&c](auto... f) { c.devData.setDependent(f.value...); }, make_tuple(DependentFields{}));
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
    void sync(DomainType& domain, ParticleDataType& simData) override
    {
        auto& d = simData.hydro;
        auto& c = simData.clust;
        auto scratchBuffers = std::tie(
            get<"ax">(d),
            get<"ay">(d),
            get<"az">(d),
            get<"rho">(d),
            get<"du">(d),
            get<"p">(d)
        );
        domain.sync(get<"keys">(d), get<"x">(d), get<"y">(d), get<"z">(d), get<"h">(d),
                    std::tie(get<"m">(d)), scratchBuffers);
        d.treeView = domain.octreeProperties();
    }

    void findClusters(DomainType& domain, ParticleDataType& simData) override
    {   
        timer.start();
        auto& d = simData.hydro;
        auto& c = simData.clust;
        c.resizeAcc(domain.nParticlesWithHalos());
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();
        c.numParticlesHalos = domain.nParticlesWithHalos();

        computeGroups(first, last, d, domain.box(), groups_);
        timer.step("Grouping particles.");

        computeLocalClusterId(
            groups_.view(),
            d,
            c,
            domain.box(),
            this->rank_,
            domain
        );
        
        timer.step("computeLocalClusters");

        if (this->numRanks_ > 1)
        {
            domain.exchangeHalos(std::tie(get<"globalClusterKeys">(c)), get<"ax">(d), get<"ay">(d));
            timer.step("mpi::synchronizeHalos");
            computeGlobalClusterId(
                                d,
                                c,
                                domain,
                                this->getRank()
                            );
            timer.step("computeGlobalClusters");
        }

        computeCompactClusterId(
            c,
            domain,
            this->getRank(),
            this->numRanks_
        );
        timer.step("compactClusterIds");

        /*
        h.resize(c.getNumClusters());
        
        reassignHalos(
            c,
            h
        );
        
        computeHaloProperties(
            domain.startIndex(),
            domain.endIndex(),
            d,
            c,
            h
        );
        timer.step("computeHaloProperties");
        */
    }

    void saveFields(IFileWriter* writer, size_t first, size_t last, ParticleDataType& simData,
                    const cstone::Box<T>& /*box*/) override
    {
        Base::outputClusterFields(writer, first, last, simData);
        timer.step("FileOutput");
    }
};

} // namespace sphexa