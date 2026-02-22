/*
 * MIT License
 *
 * SPH-EXA Standalone Clustering Tool
 * Copyright (c) 2024 CSCS, ETH Zurich, University of Basel, University of Zurich
 *
 * Standalone clustering program for post-processing SPH simulation data
 * 
 * @author Vincente Della Balda
 */

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "cluster/hdf5_data.hpp"
#include "cstone/domain/domain.hpp"

#include "init/factory.hpp"
#include "io/arg_parser.hpp"
#include "io/factory.hpp"
#include "io/cluster_hdf5_writer.hpp"
#include "propagator/factory.hpp"
#include "factory.hpp"
#include "sph/types.hpp"
#include "util/timer.hpp"
#include "util/pm_reader.hpp"
#include "util/utils.hpp"

#include "sphexa/simulation_data.hpp"

#ifdef USE_CUDA
using AccType = cstone::GpuTag;
#else
using AccType = cstone::CpuTag;
#endif

namespace fs = std::filesystem;
using namespace sphexa;

void printHelp(char* binName, int rank);
int getNumLocalRanks(int);

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

    // Required parameters
    const std::string        inputFile    = parser.get("--input");
    if (inputFile.empty()) {
        if (rank == 0) std::cerr << "Error: --input parameter is required\n";
        printHelp(argv[0], rank);
        return EXIT_FAILURE;
    }

    // Clustering parameters
    const double             b                        = parser.get("--percolation-factor", 0.2);
    const double             percolationLengthDefault = parser.get("--percolation-length", 0.0);
    const float              mergeFactor              = parser.get("--merge-factor", 0.3f);
    const int                clusterThreshold         = parser.get("--cluster-threshold", 64);
    const bool               haloProp                 = parser.exists("--halo-prop");
    const bool               findSubclusters          = parser.exists("--subcluster");
    const bool               sortByCluster            = parser.exists("--sort-by-cluster");
    const std::string        clustChoice              = "dark";

    if (findSubclusters && numRanks > 1) {
        if (rank == 0) std::cerr << "Error: Subclustering is not supported in parallel yet\n";
        printHelp(argv[0], rank);
        return EXIT_FAILURE;
    }

    // Output parameters
    std::string              outFile      = parser.get("-o", removeModifiers(inputFile));
    const bool               ascii        = parser.exists("--ascii");
    const bool               quiet        = parser.exists("--quiet");
    const bool               avClean      = parser.exists("--avclean");
    const bool               profEnabled  = parser.exists("--profile");
    const std::string        pmroot       = parser.get("--pmroot", std::string("")); // /sys/cray/pm_counters
    std::string              profFile     = parser.get("-op", std::string("profile"));
    std::vector<std::string> outputFields = parser.getCommaList("-f");

    std::ofstream nullOutput("/dev/null");
    std::ostream& output = (quiet || rank) ? nullOutput : std::cout;

    // Create file I/O objects
    auto fileWriter = fileWriterFactory(ascii, MPI_COMM_WORLD);
    auto fileReader = fileReaderFactory(ascii, MPI_COMM_WORLD);
    std::string glassBlock;
    auto simInit    = initializerFactory<Dataset>(inputFile, glassBlock, fileReader.get());

    // Create clusterer
    auto clusterer = clustFactory<Domain, Dataset>(clustChoice, findSubclusters, output, rank);

    // Create propagator
    std::string propChoice = "ve"; // dummy propagator for loading data, not used for actual time integration
    auto propagator  = propagatorFactory<Domain, Dataset>(propChoice, avClean, output, rank, simInit->constants());

    Dataset simData;
    simData.comm = MPI_COMM_WORLD;

    Timer totalTimer(output);
    MPI_Barrier(MPI_COMM_WORLD);
    totalTimer.start();

    if (rank == 0) {
        std::cout << "=== SPH-EXA Standalone Clustering Tool ===\n";
        std::cout << "Input file: " << inputFile << "\n";
        std::cout << "Percolation factor: " << b << "\n";
        std::cout << "Cluster threshold: " << clusterThreshold << "\n";
        if (percolationLengthDefault > 0.0) {
            std::cout << "Fixed percolation length: " << percolationLengthDefault << "\n";
        }
        if (findSubclusters)
        {
            std::cout << "Subclustering enabled with merge factor " << mergeFactor << "\n";
        }
        if (sortByCluster) {
            std::cout << "Cluster-sorted output enabled\n";
        }
        std::cout << "Output file: " << outFile << "\n";
    }

    // Load initial data
    propagator->activateFields(simData);
    propagator->load(inputFile, fileReader.get());
    auto box = simInit->init(rank, numRanks, 50, simData, fileReader.get());
   
    auto& d = simData.hydro;
    auto& c = simData.clust;
    auto& h = simData.halo;

    if (rank == 0) {
        std::cout << "Loaded " << d.numParticlesGlobal << " particles from " << inputFile << "\n";
    }

    // Set output fields
    if (outputFields.empty()) {
        outputFields = {"halo_id", "id"};
    }
    simData.setOutputFields(outputFields);
    std::vector<std::string> clusterOutputFiels =
            {"cId", "globalSize", "cMass", "xCenter", "yCenter", "zCenter", "xVelocity", "yVelocity", "zVelocity"};
    simData.setOutputFields(clusterOutputFiels);

    // Activate clustering fields
    clusterer->addCounters(pmroot, getNumLocalRanks(numRanks));
    clusterer->activateFields(simData);
    clusterer->setNumRanks(numRanks);

    double percolationLength;
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

    auto haloComm = MPI_COMM_SELF;
    std::unique_ptr<IFileWriter> haloWriter;    
    if (rank == 0) {
        haloWriter = fileWriterFactory(ascii, haloComm);
        std::cout << "Percolation length: " << percolationLength << "\n";
    }

    // Set up domain for clustering
    uint64_t bucketSizeFocus = 64;
    uint64_t bucketSize = std::max(bucketSizeFocus, d.numParticlesGlobal / (100 * numRanks));
    float theta = 1.0f; // No gravity needed for clustering
    
    Domain domain(rank, numRanks, bucketSize, bucketSizeFocus, theta, box);
    domain.setGrowthAllocRate(simData.hydro.getAllocGrowthRate());
    domain.setPercLength(percolationLength);

    // Perform clustering
    clusterer->sync(domain, simData);
    clusterer->findClusters(domain, simData);
    
    if (sortByCluster) {    
        std::string clusterOutFile = outFile + "_cluster.h5";
        ClusterHDF5Writer hdf5Writer(MPI_COMM_WORLD);
        std::vector<std::string> outFieldNames = {"x", "y", "z", "vx", "vy", "vz", "m", "halo_id", "id"};
        
        cluster::HDF5Data<AccType> hdf5Data;
        hdf5Data.activateFields(outFieldNames);
        hdf5Data.gatherFields(outFieldNames, simData, simData.halo.particleToHaloMap);
        auto hostData = hdf5Data.getHostData(outFieldNames);
        auto clusterInfos = hdf5Data.createClusterInfos(simData.halo);
        
        hdf5Writer.createFile(clusterOutFile, h.nClusteredGlobal, clusterInfos, outFieldNames);
        hdf5Writer.writeParticles(outFieldNames, hostData, clusterInfos);
        hdf5Writer.writeClusterMetadata(clusterInfos);
        hdf5Writer.close();
        if (rank == 0) {
            std::cout << "Cluster-sorted output written to: " << clusterOutFile << "\n";
        }
    }

    if (haloProp)
    {
        clusterer->computeHaloProperties(domain, simData);
    }

    if (findSubclusters)
    {
        clusterer->findSubClusters(domain, simData);
    }
    
    Timer writeTimer(output);
    writeTimer.start();
    
    fileWriter->addStep(domain.startIndex(), domain.endIndex(), outFile+fileWriter->suffix());
    simData.hydro.loadOrStoreAttributes(fileWriter.get());
    box.loadOrStore(fileWriter.get());
    
    // Write clustering-specific data
    simData.clust.loadOrStoreAttributes(fileWriter.get());
    propagator->saveFields(fileWriter.get(), domain.startIndex(), domain.endIndex(), simData, box);
    clusterer->saveFields(fileWriter.get(), domain.startIndex(), domain.endIndex(), simData, box);
    clusterer->save(fileWriter.get());
    propagator->save(fileWriter.get());

    fileWriter->closeStep();
 
    //Write halo properties to separate file (rank 0 only)
    // Communicator of only rank 
    if (rank == 0 && haloProp)
    {
        std::string haloFile = outFile + "_halos" + haloWriter->suffix();
        haloWriter->addStep(0, h.getNumClustersGlobal(), haloFile);
        clusterer->writeHaloProperties(haloFile, simData, haloWriter.get());
        haloWriter->closeStep();
    }

    auto fileWriterSeq = fileWriterFactory(ascii, MPI_COMM_WORLD, true);
    if (profEnabled) { clusterer->writeMetrics(fileWriterSeq.get(), profFile); }
    
    writeTimer.step("Output written");
    totalTimer.step("Total execution time");

    if (rank == 0) {
        std::cout << "Clustering completed successfully!\n";
        std::cout << "Results written to: " << outFile << "\n";
    }

    return exitSuccess();
}

void printHelp(char* name, int rank)
{
    if (rank == 0)
    {
        printf("\n=== SPH-EXA Standalone Clustering Tool ===\n\n");
        printf("Usage: %s --input INPUT_FILE [OPTIONS]\n\n", name);
        printf("Required Arguments:\n");
        printf("  --input FILE\t\tInput HDF5 file with particle data\n\n");
        
        printf("Clustering Options:\n");
        printf("  --percolation-factor NUM\tPercolation factor for automatic length calculation [0.2]\n");
        printf("  --percolation-length NUM\tFixed percolation length (overrides factor calculation) [auto]\n");
        printf("  --cluster-threshold NUM\tMinimum particles per cluster [64]\n\n");
        
        printf("Output Options:\n");
        printf("  -o FILE\t\tOutput file name [clusters_INPUT_FILE]\n");
        printf("  --ascii\t\tWrite ASCII format instead of HDF5 [binary HDF5]\n");
        printf("  -f LIST\t\tComma-separated list of fields to write [x,y,z,h,m,halo_id]\n\n");
        
        printf("General Options:\n");
        printf("  --quiet\t\tSuppress output messages\n");
        printf("  --avclean\t\tUse cleaned averaging in clustering\n");
        printf("  -h, --help\t\tShow this help message\n\n");
        
        printf("Examples:\n");
        printf("  %s --input simulation_output.h5\n", name);
        printf("  %s --input data.h5 -o my_clusters.h5 --percolation-length 0.1\n", name);
        printf("  %s --input data.h5 --cluster-threshold 32 --ascii\n", name);
        printf("\n");
    }
}

int getNumLocalRanks(int defValue)
{
    return getenv("SLURM_NTASKS_PER_NODE") == nullptr ? defValue : std::stoi(getenv("SLURM_NTASKS_PER_NODE"));
}