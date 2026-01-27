/*! @file
 * @brief An interface for different types of clusterers
 *
 * @author Vincente Della Balda vinc.dellabalda@gmail.com>
 */

#pragma once

#include <variant>

#include "cstone/sfc/box.hpp"
#include "io/ifile_io.hpp"
#include "sph/particles_data.hpp"
#include "util/pm_reader.hpp"
#include "util/timer.hpp"


namespace sphexa
{

template<class DomainType, class ParticleDataType>
class Clusterer
{
    using T              = typename ParticleDataType::RealType;
    using KeyType        = typename ParticleDataType::KeyType;


public:
    Clusterer(std::ostream& output, int rank)
        : out(output)
        , timer(output)
        , pmReader(rank)
        , rank_(rank)
    {
    }

    //! @brief whether conserved quantities are time-synchronized (when completing a full time-step hierarchy)
    virtual bool isSynced() { return true; }

    //! @brief add pm counters if they exist
    void addCounters(const std::string& pmRoot, int numRanksPerNode) { pmReader.addCounters(pmRoot, numRanksPerNode); }

    //! @brief print timing info
    void writeMetrics(IFileWriter* writer, const std::string& outFile)
    {
        timer.writeTimings(writer, outFile);
        pmReader.writeTimings(writer, outFile);
    };

    int getRank() { return rank_; }
    void setNumRanks(int numRanks) { numRanks_ = numRanks; }
    int getNumRanks() { return numRanks_; }
    
    //! @brief get a list of field strings marked as conserved at runtime
    virtual std::vector<std::string> conservedFields() const = 0;

    //! @brief Marks conserved and dependent fields inside the particle dataset as active, enabling memory allocation
    virtual void activateFields(ParticleDataType& d) = 0;

    //! @brief save particle data fields to file
    virtual void saveFields(IFileWriter*, size_t, size_t, ParticleDataType&, const cstone::Box<T>&){};

    //! @brief save internal state to file
    virtual void save(IFileWriter*) {}

    //! @brief load internal state from file
    virtual void load(const std::string& path, IFileReader*) {}

    //! @brief synchronize computational domain
    virtual void sync(DomainType& domain, ParticleDataType& d){};

    virtual void findClusters(DomainType& domain, ParticleDataType& d){};

    virtual ~Clusterer() = default;

protected:
    static void outputClusterFields(IFileWriter* writer, ParticleDataType& simData)
    {
        auto output = [](auto& d, IFileWriter* writer)
        {
            auto fieldPointers = d.data();
            auto indicesDone   = d.outputFieldIndices;
            auto namesDone     = d.outputFieldNames;

            for (int i = int(indicesDone.size()) - 1; i >= 0; --i)
            {
                int fidx = indicesDone[i];
                if (d.isAllocated(fidx))
                {
                    int column = std::find(d.outputFieldIndices.begin(), d.outputFieldIndices.end(), fidx) -
                                 d.outputFieldIndices.begin();
                    std::visit(
                        [writer, c = column, key = namesDone[i]](auto field)
                        {
                            auto&& tmp = toHost(*field);
                            writeField(writer, key, tmp.data(), c);
                        },
                        fieldPointers[fidx]);
                    indicesDone.erase(indicesDone.begin() + i);
                    namesDone.erase(namesDone.begin() + i);
                }
            }

            if (!indicesDone.empty() && writer->rank() == 0)
            {
                std::cout << "WARNING: the following fields are not in use and therefore not output: ";
                for (int fidx = 0; fidx < indicesDone.size() - 1; ++fidx)
                {
                    std::cout << d.fieldNames[fidx] << ",";
                }
                std::cout << d.fieldNames[indicesDone.back()] << std::endl;
            }
        };

        output(simData.clust, writer);
    }

    std::ostream& out;
    Timer         timer;
    PmReader      pmReader;
    int           rank_;
    int           numRanks_;
};

} // namespace sphexa