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

#include "cstone/domain/domain.hpp"

#include "init/factory.hpp"
#include "io/arg_parser.hpp"
#include "io/factory.hpp"
#include "propagator/factory.hpp"
#include "factory.hpp"
#include "sph/types.hpp"
#include "util/timer.hpp"
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
    const int                clusterThreshold         = parser.get("--cluster-threshold", 64);
    const std::string        clustChoice              = "dark";

    // Output parameters
    std::string              outFile      = parser.get("-o", removeModifiers(inputFile));
    const bool               ascii        = parser.exists("--ascii");
    const bool               quiet        = parser.exists("--quiet");
    const bool               avClean      = parser.exists("--avclean");
    std::vector<std::string> outputFields = parser.getCommaList("-f");

    std::ofstream nullOutput("/dev/null");
    std::ostream& output = (quiet || rank) ? nullOutput : std::cout;

    // Create file I/O objects
    auto fileWriter = fileWriterFactory(ascii, MPI_COMM_WORLD);
    auto fileReader = fileReaderFactory(ascii, MPI_COMM_WORLD);
    std::string glassBlock;
    auto simInit    = initializerFactory<Dataset>(inputFile, glassBlock, fileReader.get());

    // Create clusterer
    auto clusterer = clustFactory<Domain, Dataset>(clustChoice, avClean, output, rank);

    // Create propagator
    std::string propChoice = "ve";
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
        std::cout << "Output file: " << outFile << "\n";
    }

    // Load initial data
    propagator->activateFields(simData);
    propagator->load(inputFile, fileReader.get());
    auto box = simInit->init(rank, numRanks, 50, simData, fileReader.get());
   
    auto& d = simData.hydro;
    auto& c = simData.clust;


    if (rank == 0) {
        std::cout << "Loaded " << d.numParticlesGlobal << " particles from " << inputFile << "\n";
    }

    // Set output fields
    if (outputFields.empty()) {
        outputFields = {"x", "y", "z", "halo_id"};
    }
    simData.setOutputFields(outputFields);

    // Activate clustering fields
    clusterer->activateFields(simData);
    clusterer->setNumRanks(numRanks);

    // Calculate percolation length
    double percolationLength;
    if (percolationLengthDefault > 0.0) {
        percolationLength = percolationLengthDefault;
    } else {
        double simulationVolume = box.lx() * box.ly() * box.lz();
        double meanInterparticleSeparation = std::pow(simulationVolume / d.numParticlesGlobal, 1.0/3.0);
        percolationLength = b * meanInterparticleSeparation;
    }

    // Set up clustering parameters
    c.setPercLength(percolationLength);
    c.numParticlesGlobal = d.numParticlesGlobal;
    c.clusterThreshold = clusterThreshold;

    if (rank == 0) {
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
    Timer clusterTimer(output);
    clusterTimer.start();
    
    clusterer->sync(domain, simData);
    clusterTimer.step("Domain synchronized");

    clusterer->findClusters(domain, simData);    
    clusterTimer.step("Clustering completed");

    // Write results
    if (!parser.exists("-o")) { 
        outFile += fileWriter->suffix(); 
    }
    
    Timer writeTimer(output);
    writeTimer.start();
    
    // Write particle data with cluster assignments
    fileWriter->addStep(domain.startIndex(), domain.endIndex(), "cluster_"+outFile);
    simData.hydro.loadOrStoreAttributes(fileWriter.get());
    box.loadOrStore(fileWriter.get());
    
    // Write clustering-specific data
    simData.clust.loadOrStoreAttributes(fileWriter.get());
    clusterer->saveFields(fileWriter.get(), domain.startIndex(), domain.endIndex(), simData, box);
    clusterer->save(fileWriter.get());
    
    fileWriter->closeStep();
    
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