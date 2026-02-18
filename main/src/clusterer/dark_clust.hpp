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
#include "sph/hydro_std/density.hpp"

#include "cluster/cluster.hpp"
#include "cluster/cluster_data.hpp"
#include "cluster/cluster_tree.hpp"
#include "cluster/densmax.hpp"
#include "cluster/halo.hpp"
#include "cluster/halo_data.hpp"

#include "iclusterer.hpp"

namespace sphexa
{

using namespace sph;
using namespace cluster;
using util::FieldList;

template<class DomainType, class ParticleDataType>
class darkClust : public Clusterer<DomainType, ParticleDataType>
{
protected:
    using Base = Clusterer<DomainType, ParticleDataType>;
    using Base::pmReader;
    using Base::timer;

    using T              = typename ParticleDataType::RealType;
    using KeyType        = typename ParticleDataType::KeyType;

    using Acc       = typename ParticleDataType::AcceleratorType;

    GroupData<Acc> groups_;

    using ConservedFields = FieldList<"halo_id", "sub_id", "work_id", "hTight">;

    using DependentFields =
        FieldList<"flagged", "idBuf", "keyBuf", "localClusterKeys", "globalClusterKeys", "candidateDensity", "candidateZone">;


public:
    darkClust(std::ostream& output, size_t rank, bool avClean)
        : Base(output, rank)
    {
    }

    std::vector<std::string> conservedFields() const override
    {
        std::vector<std::string> ret{};
        for_each_tuple([&ret](auto f) { ret.push_back(f.value); }, make_tuple(ConservedFields{}));
        return ret;
    }

    void activateFields(ParticleDataType& simData) override
    {
        auto& c = simData.clust;
        std::apply([&c](auto... f) { c.setConserved(f.value...); }, make_tuple(ConservedFields{}));
        std::apply([&c](auto... f) { c.setDependent(f.value...); }, make_tuple(DependentFields{}));

        auto&h = simData.halo;
        h.setConserved("id", "globalSize", "mass", "xCenter", "yCenter", "zCenter", "xVelocity", "yVelocity", "zVelocity");
        h.setDependent("localSize", "localOffset", "globalOffset");

        auto& d = simData.hydro;
        d.setConserved("x", "y", "z", "vx", "vy", "vz", "m", "id", "h");
        d.setDependent("rho", "p", "c", "ax", "ay", "az", "du", "c11", "c12", "c13", "c22", "c23", "c33", "nc");
    }
    
    void sync(DomainType& domain, ParticleDataType& simData) override
    {
        auto& d = simData.hydro;
        auto& c = simData.clust;
        auto conserved = std::tie(get<"id">(d), get<"vx">(d), get<"vy">(d), get<"vz">(d));
        auto scratchBuffers = std::tie(
            get<"ax">(d), get<"ay">(d), get<"az">(d), get<"rho">(d), get<"p">(d), get<"c">(d),
            get<"du">(d), get<"c11">(d), get<"c12">(d), get<"c13">(d), get<"c22">(d), get<"c23">(d),
            get<"c33">(d), get<"nc">(d), get<"flagged">(c), get<"idBuf">(c), get<"keyBuf">(c));
        domain.sync(get<"keys">(d), get<"x">(d), get<"y">(d), get<"z">(d), get<"h">(d),
                    std::tuple_cat(std::tie(get<"m">(d)), conserved),
                    scratchBuffers);
        d.treeView = domain.octreeProperties();
    }

    void findClusters(DomainType& domain, ParticleDataType& simData) override
    {   
        timer.start();
        pmReader.start();

        auto& d = simData.hydro;
        auto& c = simData.clust;
        auto& h = simData.halo;
        c.resize(domain.nParticlesWithHalos());
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();
        c.numParticles = domain.nParticles();
        c.numParticlesHalos = domain.nParticlesWithHalos();

        computeGroups(first, last, d, domain.box(), groups_);
        timer.step("Grouping particles.");

        computeLocalClusterId(groups_.view(), d, c, domain.box(), domain);
        timer.step("computeLocalClusters");
        pmReader.step();

        if (this->numRanks_ > 1)
        {
            domain.exchangeHalos(std::tie(get<"globalClusterKeys">(c)), get<"ax">(d), get<"ay">(d));
            timer.step("mpi::synchronizeHalos");
            computeGlobalClusterId(d, c, domain);
            timer.step("computeGlobalClusters");
            pmReader.step();
        }

        computeCompactClusterId(c, h, domain);
        timer.step("compactClusterIds");
        pmReader.step();

        h.resize(c.numClustersGlobal);
        prepareParticleClusterMap(c, h, domain);
        timer.step("prepareParticleClusterMap");
        pmReader.step();
    }

    void findSubClusters(
        DomainType& domain,
        ParticleDataType& simData
    )
    {;
        auto& d = simData.hydro;
        auto& c = simData.clust;
        auto & h = simData.halo;
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        findNeighborsSfc(first, last, d, domain.box());
        computeGroups(first, last, d, domain.box(), groups_);
        timer.step("FindNeighbors::subcluster");
        pmReader.step();

        computeDensity(groups_.view(), d, domain.box());
        timer.step("Density::subcluster");
        pmReader.step();

        domain.exchangeHalos(std::tie(get<"rho">(d), get<"halo_id">(c)), get<"ax">(d), get<"ay">(d));
        timer.step("mpi::synchronizeHalos");

        computeDensityGroups(groups_.view(), d, c, domain.box());
        timer.step("DensityGroups::subcluster");
        pmReader.step();

        if (this->numRanks_ > 1)
        {
            domain.exchangeHalos(std::tie(get<"globalClusterKeys">(c)), get<"ax">(d), get<"ay">(d));
            timer.step("mpi::synchronizeHalos");
            computeGlobalClusterId(d, c, domain);
            timer.step("computeGlobalClusterId::subcluster");
            pmReader.step();
        }

        computeCompactClusterId(c, h, domain);
        timer.step("compactClusterId::subcluster");
        pmReader.step();
    }

    void computeHaloProperties(
        DomainType& domain,
        ParticleDataType& simData)
    {
        timer.start();
        auto& d = simData.hydro;
        auto& c = simData.clust;
        auto& h = simData.halo;
        
        computeHaloPropertiesLocal(
            domain.startIndex(),
            domain.endIndex(),
            d,
            c,
            h
        );
        timer.step("computeHaloPropertiesLocal");

        communicateHaloProperties(
            domain.startIndex(),
            domain.endIndex(),
            h
        );
        timer.step("communicateHaloProperties");
    }

    void saveFields(IFileWriter* writer, size_t first, size_t last, ParticleDataType& simData,
                    const cstone::Box<T>& /*box*/) override
    {
        Base::outputClusterFields(writer, simData);
        timer.step("FileOutput");
    }

    void writeHaloProperties(const std::string& filename, ParticleDataType& simData, IFileWriter* writer) override
    {
        if (this->rank_ != 0) return; // Only rank 0 writes
        
        timer.start();
        auto& h = simData.halo;

        uint32_t totalNumHalos = h.getNumClustersGlobal();
        
        if (totalNumHalos == 0) {
            if (this->rank_ == 0) {
                this->out << "No halos found, skipping halo output." << std::endl;
            }
            return;
        }
        
        // Set halo fields for output
        std::vector<std::string> outFields = 
            {"id", "globalSize", "mass", "xCenter", "yCenter", "zCenter", "xVelocity", "yVelocity", "zVelocity"};
        h.setOutputFields(outFields);
        
        // Add step to HDF5 file for halo output
        writer->addStep(0, totalNumHalos, filename);
        
        // Use the existing field infrastructure to write data
        auto fieldPointers = h.data();
        for (int i = 0; i < h.outputFieldIndices.size(); ++i) {
            int fidx = h.outputFieldIndices[i];
            if (h.isAllocated(fidx)) {
                const std::string& fieldName = h.outputFieldNames[i];
                std::visit([&](auto* fieldPtr) {
                    auto hostData = toHost(*fieldPtr);
                    writeField(writer, fieldName, hostData.data(), 0);
                }, fieldPointers[fidx]);
            }
        }
        
        writer->closeStep();
        
        timer.step("Halo properties output");
        
        printf("Halo properties written to: %s\n", filename.c_str());
        printf("Total halos: %u\n", totalNumHalos);
    }
};

} // namespace sphexa