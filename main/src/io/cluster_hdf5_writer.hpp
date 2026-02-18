/*
 * MIT License
 * 
 * Custom HDF5 writer for cluster-sorted particle data
 * Bypasses H5hut for direct HDF5 control over file structure
 */

#ifndef CLUSTER_HDF5_WRITER_HPP
#define CLUSTER_HDF5_WRITER_HPP

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <mpi.h>
#include <hdf5.h>

#include "cluster/hdf5_data_utils.hpp"

namespace sphexa {

class ClusterHDF5Writer {
public:
    explicit ClusterHDF5Writer(MPI_Comm comm);
    ~ClusterHDF5Writer();

    using ClusterInfo = cluster::ClusterInfo;
    using HDF5HostData = cluster::HDF5HostData;
    
    // Disable copy/move for simplicity
    ClusterHDF5Writer(const ClusterHDF5Writer&) = delete;
    ClusterHDF5Writer& operator=(const ClusterHDF5Writer&) = delete;
    
    /**
     * @brief Create HDF5 file with proper structure for cluster data
     * @param filename Output filename
     * @param totalParticles Total number of classified particles globally
     * @param clusterInfos Information about all clusters
     * @param fieldNames List of particle fields to write (e.g. "x", "y", "z", "vx", "vy", "vz", "m", "halo_id", "id")
     */
    void createFile(const std::string& filename, 
                   size_t totalParticles,
                   const std::vector<ClusterInfo>& clusterInfos,
                   const std::vector<std::string>& fieldNames);

    /**
     * @brief Write sorted particle data to HDF5 file
     * @param fieldNames List of particle fields to write (must match those used in createFile)
     * @param particles Particle data sorted by cluster (host data only)
     * @param clusterInfos Cluster layout information
     */
    void writeParticles(const std::vector<std::string>& fieldNames,
                       const HDF5HostData& particles, 
                       const std::vector<ClusterInfo>& clusterInfos);

    /**
     * @brief Write cluster metadata (cluster properties, statistics)
     * @param clusterInfos Cluster information
     */
    void writeClusterMetadata(const std::vector<ClusterInfo>& clusterInfos);

    /**
     * @brief Close the HDF5 file
     */
    void close();

private:
    MPI_Comm comm_;
    int rank_, numRanks_;    
    hid_t file_id_;
    hid_t particles_group_id_;
    hid_t clusters_group_id_;
    std::string filename_;
    
    // Helper methods
    hid_t createGroup(hid_t parent_id, const std::string& name);
    hid_t createDataset(hid_t group_id, const std::string& name, 
                       hid_t datatype, size_t total_size); 
   
    template<typename T>
    void writeField(hid_t group_id, const std::string& name, 
                   const std::vector<T>& data, size_t total_size,
                   size_t global_offset);

    template<typename T>
    hid_t getHDF5Type();
};

} // namespace sphexa

#endif // CLUSTER_HDF5_WRITER_HPP