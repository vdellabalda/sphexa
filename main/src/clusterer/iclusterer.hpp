/*
 * MIT License
 *
 * SPH-EXA
 * Copyright (c) 2024 CSCS, ETH Zurich, University of Basel, University of Zurich
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
 * @brief Interface for clustering particles into groups
 *
 * @author Vincente Della Balda vinc.dellabalda@gmail.com>
 */

#pragma once

#include <variant>

#include "cstone/primitives/accel_switch.hpp"
#include "io/ifile_io.hpp"
#include "util/pm_reader.hpp"
#include "util/timer.hpp"

namespace sphexa
{

template<class DomainType, class ClusterDataType>
class Clusterer
{
    using T = typename ClusterDataType::KeyType;

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

    //! @brief save particle data fields to file
    virtual void saveFields(IFileWriter*, size_t, size_t, ParticleDataType&, const cstone::Box<T>&){};

    void findClusters(DomainType& domain, ParticleDataType& d, double percolationLength, int numRanks){};

    virtual ~Clusterer() = default;

protected:
    static void outputClusterFields(IFileWriter* writer, size_t first, size_t last, ParticleDataType& simData)
    {
        auto output = [](size_t first, size_t last, auto& d, IFileWriter* writer)
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
                    transferToHost(d, first, last, {d.fieldNames[fidx]});
                    std::visit([writer, c = column, key = namesDone[i]](auto field)
                               { writer->writeField(key, field->data(), c); },
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

        output(first, last, simData.clust, writer);
    }

    std::ostream& out;
    Timer         timer;
    PmReader      pmReader;
    int           rank_;
};

} // namespace sphexa