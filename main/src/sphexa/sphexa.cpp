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
 * @brief SPH-EXA application front-end and main function
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 * @author Aurelien Cavelan
 * @author Jose A. Escartin <ja.escartin@gmail.com>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "cstone/domain/domain.hpp"

#include "init/factory.hpp"
#include "io/arg_parser.hpp"
#include "io/cluster_hdf5_writer.hpp"
#include "io/factory.hpp"
#include "observables/factory.hpp"
#include "propagator/factory.hpp"
#include "clusterer/factory.hpp"
#include "cluster/hdf5_data.hpp"

#include "sph/types.hpp"
#include "util/timer.hpp"
#include "util/utils.hpp"

#include "simulation_data.hpp"
#include "insitu_viz.h"

#ifdef USE_CUDA
using AccType = cstone::GpuTag;
#else
using AccType = cstone::CpuTag;
#endif

namespace fs = std::filesystem;
using namespace sphexa;

bool stopConditionReached(size_t iteration, double time, const std::string& maxStepStr);
bool syncedWallClockElapsed(float totalTimeElapsed, float wallClockLimit, float dt);
void printHelp(char* binName, int rank);
int  getNumLocalRanks(int);

int main(int argc, char** argv)
{
    auto [rank, numRanks] = initMpi();
    const ArgParser parser(argc, (const char**)argv);

    if (parser.exists("-h") || parser.exists("--h") || parser.exists("-help") || parser.exists("--help"))
    {
        printHelp(argv[0], rank);
        return exitSuccess();
    }

    using Dataset = SimulationData<AccType>;
    using Domain  = cstone::Domain<sph::SphTypes::KeyType, sph::SphTypes::CoordinateType, AccType>;

    const std::string        initCond     = parser.get("--init");
    const size_t             problemSize  = parser.get("-n", 50);
    const std::string        glassBlock   = parser.get("--glass");
    const std::string        propChoice   = parser.get("--prop", std::string("ve"));
    const std::string        maxStepStr   = parser.get("-s", std::string("200"));
    std::vector<std::string> writeExtra   = parser.getCommaList("--wextra");
    std::vector<std::string> outputFields = parser.getCommaList("-f");
    const bool               ascii        = parser.exists("--ascii");
    const bool               quiet        = parser.exists("--quiet");
    const bool               avClean      = parser.exists("--avclean");
    const int                simDuration  = parser.get("--duration", std::numeric_limits<int>::max());
    const std::string        writeFreqStr = parser.get("-w", std::string("0"));
    const bool               writeEnabled = writeFreqStr != "0" || !writeExtra.empty();
    const std::string        profFreqStr  = parser.get("--profile", maxStepStr);
    const bool               profEnabled  = parser.exists("--profile") || writeEnabled;
    const std::string        pmroot       = parser.get("--pmroot", std::string("")); // /sys/cray/pm_counters
    std::string              outFile      = parser.get("-o", "dump_" + removeModifiers(initCond));
    std::string              profFile     = parser.get("-op", std::string("profile"));

    const bool               findClusters             = parser.exists("--find-clusters");
    const double             b                        = parser.get("--percolation-factor", 0.2);
    const double             percolationLengthDefault = parser.get("--percolation-length", 0.0);
    const float              mergeFactor              = parser.get("--merge-factor", 0.3f);
    const int                clusterThreshold         = parser.get("--cluster-threshold", 64);
    const bool               findSubclusters          = parser.exists("--subcluster");
    const bool               sortByCluster            = parser.exists("--sort-by-cluster");
    const bool               haloProp                 = parser.exists("--halo-prop");
    const std::string        clustChoice              = "dark";
    std::string              clustOutFile             = outFile + "_cluster";
    std::string              clustPropOutFile         = outFile + "_cluster_properties";

    std::ofstream nullOutput("/dev/null");
    std::ostream& output = (quiet || rank) ? nullOutput : std::cout;
    std::ofstream constantsFile(fs::path(outFile).parent_path() / fs::path("constants.txt"));

    //! @brief evaluate user choice for different kind of actions
    auto fileWriter  = fileWriterFactory(ascii, MPI_COMM_WORLD);
    auto fileReader  = fileReaderFactory(ascii, MPI_COMM_WORLD);
    auto simInit     = initializerFactory<Dataset>(initCond, glassBlock, fileReader.get());
    auto propagator  = propagatorFactory<Domain, Dataset>(propChoice, avClean, output, rank, simInit->constants());
    auto observables = observablesFactory<Dataset>(simInit->constants(), constantsFile);
    std::unique_ptr<Clusterer<Domain, Dataset>> clusterer;

    Dataset simData;
    simData.comm = MPI_COMM_WORLD;

    Timer totalTimer(output);
    MPI_Barrier(MPI_COMM_WORLD);
    totalTimer.start();

    propagator->addCounters(pmroot, getNumLocalRanks(numRanks));
    propagator->activateFields(simData);
    propagator->load(initCond, fileReader.get());
    auto box = simInit->init(rank, numRanks, problemSize, simData, fileReader.get());

    auto& d = simData.hydro;
    auto& c = simData.clust;
    auto& h = simData.halo;

    simData.setOutputFields(outputFields.empty() ? propagator->conservedFields() : outputFields);
    std::vector<std::string> clusterOutputFields = {"cId", "globalSize", "cMass", "xCenter", "yCenter", "zCenter", "xVelocity", "yVelocity", "zVelocity"};
    simData.setOutputFields(clusterOutputFields);

    if (parser.exists("--G")) { d.g = parser.get<double>("--G"); }
    bool  haveGrav = (d.g != 0.0);
    float theta    = parser.get("--theta", haveGrav ? 0.5f : 1.0f);

    if (!parser.exists("-o")) { 
        outFile += fileWriter->suffix(); 
        clustOutFile += fileWriter->suffix();
        clustPropOutFile += fileWriter->suffix();
    }
    if (writeEnabled) { writeSettings(simInit->constants(), outFile, fileWriter.get()); }
    if (rank == 0) { std::cout << "Data generated for " << d.numParticlesGlobal << " global particles\n"; }

    uint64_t bucketSizeFocus = 64;
    // ~100 global nodes per rank to decompose the domain with +-1% accuracy
    uint64_t bucketSize = std::max(bucketSizeFocus, d.numParticlesGlobal / (100 * numRanks));
    Domain   domain(rank, numRanks, bucketSize, bucketSizeFocus, theta, box);
    domain.setGrowthAllocRate(simData.hydro.getAllocGrowthRate());
    domain.setPercLength(0.0);

    propagator->sync(domain, simData);
    if (rank == 0) std::cout << "Domain synchronized, nLocalParticles " << d.x.size() << std::endl;

    viz::init_catalyst(argc, argv);
    viz::init_ascent(d, domain.startIndex());

    double percolationLength;
    std::unique_ptr<IFileWriter> haloWriter;
    if (findClusters)
    {   
        clusterer = clustFactory<Domain, Dataset>(clustChoice, findSubclusters, output, rank);
        clusterer->addCounters(pmroot, getNumLocalRanks(numRanks));
        clusterer->activateFields(simData);
        clusterer->setNumRanks(numRanks);
        
        // Calculate percolation length
        if (percolationLengthDefault > 0.0)
        {
            percolationLength = percolationLengthDefault;
        }
        else
        {
            double simulationVolume = box.lx() * box.ly() * box.lz();
            double meanInterparticleSeparation = std::pow(simulationVolume / d.numParticlesGlobal, 1.0/3.0);
            percolationLength = b * meanInterparticleSeparation;
        }
        
        // Set up clustering parameters
        c.setPercLength(percolationLength);
        c.setThreshold(clusterThreshold);
        c.setMergeFactor(mergeFactor);

        if (rank == 0)
        {
            auto haloComm = MPI_COMM_SELF;
            haloWriter = fileWriterFactory(ascii, haloComm);
        }

        if (rank==0) { std::cout << "FOF clustering activated with percolation length " << percolationLength << " and cluster threshold " << clusterThreshold << std::endl;}
    }

    size_t startIteration    = d.iteration;
    bool   isOutputTriggered = true;
    
    for (bool keepRunning = true; keepRunning; d.iteration++)
    {
        domain.setPercLength((findClusters) ? percolationLength : 0.0);

        propagator->computeForces(domain, simData);
        box = domain.box();
        if (findClusters) { 
            clusterer->findClusters(domain, simData);
            if (haloProp) { clusterer->computeHaloProperties(domain, simData); }
            if (sortByCluster) { clusterer->sortByCluster(domain, simData); }
            if (findSubclusters) { clusterer->findSubClusters(domain, simData); }
        }

        if (propagator->isSynced())
        {
            observables->computeAndWrite(simData, domain.startIndex(), domain.endIndex(), box);
        }

        bool isWallClockReached = syncedWallClockElapsed(totalTimer.elapsed(), simDuration, propagator->stepElapsed());

        isOutputTriggered =
            (isOutputStep(d.iteration, writeFreqStr) || isOutputTime(d.ttot - d.minDt, d.ttot, writeFreqStr) ||
             isExtraOutputStep(d.iteration, d.ttot - d.minDt, d.ttot, writeExtra) ||
             (isWallClockReached && writeEnabled) || isOutputTriggered) &&
            d.iteration > startIteration;

        //isOutputTriggered = true;

        if (isOutputTriggered && propagator->isSynced())
        {
            fileWriter->addStep(domain.startIndex(), domain.endIndex(), outFile);
            simData.hydro.loadOrStoreAttributes(fileWriter.get());
            box.loadOrStore(fileWriter.get());
            propagator->saveFields(fileWriter.get(), domain.startIndex(), domain.endIndex(), simData, box);
            propagator->save(fileWriter.get());
            fileWriter->closeStep();

            if (findClusters)
            {
                fileWriter->addStep(domain.startIndex(), domain.endIndex(), clustOutFile);
                simData.clust.loadOrStoreAttributes(fileWriter.get());
                clusterer->saveFields(fileWriter.get(), domain.startIndex(), domain.endIndex(), simData, box);
                clusterer->save(fileWriter.get());
                fileWriter->closeStep();

                if (rank == 0 && haloProp)
                {
                    haloWriter->addStep(0, h.getNumClustersGlobal(), clustPropOutFile);
                    clusterer->writeHaloProperties(clustPropOutFile, simData, haloWriter.get());
                    haloWriter->closeStep();
                }
                
                if (sortByCluster) {    
                    ClusterHDF5Writer hdf5Writer(MPI_COMM_WORLD);
                    std::vector<std::string> outFieldNames = {"x", "y", "z", "vx", "vy", "vz", "m", "halo_id", "id"};

                    cluster::HDF5Data<AccType> hdf5Data;
                    hdf5Data.activateFields(outFieldNames);
                    hdf5Data.gatherFields(outFieldNames, simData, simData.halo.particleToHaloMap);
                    auto hostData = hdf5Data.getHostData(outFieldNames);
                    auto clusterInfos = hdf5Data.createClusterInfos(simData.halo);

                    hdf5Writer.createFile(clustOutFile, h.nClusteredGlobal, clusterInfos, outFieldNames);
                    hdf5Writer.writeParticles(outFieldNames, hostData, clusterInfos);
                    hdf5Writer.writeClusterMetadata(clusterInfos);
                    hdf5Writer.close();   
                }           
            }


            isOutputTriggered = false;
        }

        keepRunning = not(stopConditionReached(d.iteration, d.ttot, maxStepStr) || isWallClockReached) ||
                      not propagator->isSynced();

        viz::execute(d, domain.startIndex(), domain.endIndex());

        propagator->integrate(domain, simData);
        propagator->printIterationTimings(domain, simData);

        if (isOutputStep(d.iteration, profFreqStr) || isOutputTime(d.ttot - d.minDt, d.ttot, profFreqStr) ||
            isWallClockReached)
        {
            auto fileWriterSeq = fileWriterFactory(ascii, MPI_COMM_WORLD, true);
            if (profEnabled) { propagator->writeMetrics(fileWriterSeq.get(), profFile); }
        }

    }
    totalTimer.step("Total execution time of " + std::to_string(d.iteration - startIteration) + " iterations of " +
                    initCond + " up to t = " + std::to_string(d.ttot));

    constantsFile.close();
    viz::finalize();
    return exitSuccess();
}

//! @brief check whether the stop conditions based on evolved time (not wall-clock) or iteration count are reached
bool stopConditionReached(size_t iteration, double time, const std::string& maxStepStr)
{
    bool lastIteration = strIsIntegral(maxStepStr) && iteration >= std::stoi(maxStepStr);
    bool simTimeLimit  = !strIsIntegral(maxStepStr) && time > std::stod(maxStepStr);

    return lastIteration || simTimeLimit;
}

/*! @brief check whether wall clock limit was reached on any rank
 *
 * We do this to account for the fact that the total runtime timestamp might not be taken at exactly the same moment
 * across ranks.
 */
bool syncedWallClockElapsed(float totalTimeElapsed, float wallClockLimit, float dt)
{
    // if total time elapsed is getting close to (within dt) of the limit
    if (totalTimeElapsed + dt > wallClockLimit)
    {
        int isLimitReachedAny = totalTimeElapsed > wallClockLimit;
        mpiAllreduce(MPI_IN_PLACE, &isLimitReachedAny, 1, MPI_SUM, MPI_COMM_WORLD);
        return isLimitReachedAny;
    }
    return false;
}

int getNumLocalRanks(int defValue)
{
    return getenv("SLURM_NTASKS_PER_NODE") == nullptr ? defValue : std::stoi(getenv("SLURM_NTASKS_PER_NODE"));
}

void printHelp(char* name, int rank)
{
    if (rank == 0)
    {
        printf("\nUsage:\n\n");
        printf("%s [OPTIONS]\n", name);
        printf("\nWhere possible options are:\n\n");

        printf("\t--init \t\t Test case selection (evrard, sedov, noh, isobaric-cube, wind-shock, turbulence)\n"
               "\t\t\t or an HDF5 file with initial conditions\n\n");
        printf("\t-n NUM \t\t Initialize data with (approx when using glass blocks) NUM^3 global particles [50]\n");
        printf("\t--glass FILE\t Use glass block as template to generate initial x,y,z configuration\n\n");

        printf("\t--theta NUM \t Gravity accuracy parameter [default 0.5 when self-gravity is active]\n\n");

        printf("\t--G NUM \t Gravitational constant [default dependent on test-case selection]\n\n");

        printf("\t--prop STRING \t Choice of SPH propagator [default: modern SPH]. For standard SPH, use \"std\" \n\n");

        printf("\t-s NUM \t\t int(NUM):  Number of iterations (time-steps) [200],\n\
                \t real(NUM): Time of simulation (time-model)\n\n");

        printf("\t--wextra LIST \t Comma-separated list of steps (integers) or ~times (floating point)\n"
               "\t\t\t at which to trigger file output\n"
               "\t\t\t e.g.: --wextra 1,10,0.77 (output at after iteration 1 and 10 and at simulation time 0.77s\n\n");

        printf("\t-w NUM \t\t NUM<=0:    Disable file output [default],\n\
                \t int(NUM):  Dump particle data every NUM iteration steps,\n\
                \t real(NUM): Dump particle data every NUM seconds of simulation (not wall-clock) time \n\n");

        printf("\t-f LIST \t Comma-separated list of field names to write for each dump.\n"
               "\t\t\t e.g: -f x,y,z,h,rho\n"
               "\t\t\t If omitted, the list will be set to all conserved fields,\n"
               "\t\t\t resulting in a restartable output file\n\n");

        printf("\t--ascii \t Dump file in ASCII format [binary HDF5 by default]\n\n");

        printf("\t--outDir PATH \t Path to directory where output will be saved [./].\n\
                \t Note that directory must exist and be provided with ending slash,\n\
                \t e.g: --outDir /home/user/folderToSaveOutputFiles/\n\n");

        printf("\t--quiet \t Don't print anything to stdout\n\n");

        printf("\t--duration \t Maximum wall-clock run time of the simulation in seconds.[MAX_INT]\n\n");

        printf("\t--profile \t\t Enable profiling output,\n\
                \t Profiling is enabled by default if file output is enabled.\n\n");

        printf("\t--profileFreq NUM \t\t [default]: the profiling data is outputted at the end of the simulation,\n\
                \t NUM<=0:    Disable profiling output,\n\
                \t int(NUM):  Dump profiling data every NUM iteration steps,\n\
                \t real(NUM): Dump profiling data every NUM seconds of simulation (not wall-clock) time \n\n");
    }
}
