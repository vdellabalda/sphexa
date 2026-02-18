// Create this as: /home/vincente/software/sphexa/main/src/io/cluster_hdf5_writer.cpp

#include "cluster_hdf5_writer.hpp"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cassert>
#include <stdexcept>
#include <type_traits>
#include <cstdint>

namespace sphexa {

    using ClusterInfo = cluster::ClusterInfo;
    using HDF5HostData = cluster::HDF5HostData;

inline void h5Check(herr_t status, const char* what)
{
    if (status < 0) throw std::runtime_error(std::string("HDF5 call failed: ") + what);
}

inline void h5CheckId(hid_t id, const char* what)
{
    if (id < 0) throw std::runtime_error(std::string("HDF5 id invalid: ") + what);
}

ClusterHDF5Writer::ClusterHDF5Writer(MPI_Comm comm) 
    : comm_(comm), file_id_(H5I_INVALID_HID), 
      particles_group_id_(H5I_INVALID_HID), clusters_group_id_(H5I_INVALID_HID) {
    
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &numRanks_);
}

ClusterHDF5Writer::~ClusterHDF5Writer() {
    close();
}

void ClusterHDF5Writer::createFile(const std::string& filename, 
                                  size_t totalParticles,
                                  const std::vector<ClusterInfo>& clusterInfos,
                                  const std::vector<std::string>& fieldNames) {
    filename_ = filename;

    // Use consistent communicator throughout
    // Create property list for parallel access
    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
    h5CheckId(plist_id, "H5Pcreate(H5P_FILE_ACCESS)");
    h5Check(H5Pset_fapl_mpio(plist_id, comm_, MPI_INFO_NULL), "H5Pset_fapl_mpio");

    // Make metadata operations more robust in parallel builds that support it.
    // (These calls are no-ops on older HDF5 versions.)
#if defined(H5_VERSION_GE)
#if H5_VERSION_GE(1,10,0)
    (void)H5Pset_coll_metadata_write(plist_id, 1);
    (void)H5Pset_all_coll_metadata_ops(plist_id, 1);
#endif
#endif
    
    // Create the file collectively
    file_id_ = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    h5CheckId(file_id_, "H5Fcreate");
    H5Pclose(plist_id);
    
    if (file_id_ < 0) {
        throw std::runtime_error("Failed to create HDF5 file: " + filename);
    }
    
    // Create main groups
    particles_group_id_ = createGroup(file_id_, "particles");
    clusters_group_id_ = createGroup(file_id_, "clusters");
    
    // Create datasets only for requested fields (collective across all ranks)
    // Use explicit-width types for portability.
    for (const auto& fieldName : fieldNames) {
        hid_t dataset_id = H5I_INVALID_HID;
        if (fieldName == "x" || fieldName == "y" || fieldName == "z") {
            dataset_id = createDataset(particles_group_id_, fieldName, H5T_NATIVE_DOUBLE, totalParticles);
        }
        else if (fieldName == "vx" || fieldName == "vy" || fieldName == "vz" || fieldName == "m") {
            dataset_id = createDataset(particles_group_id_, fieldName, H5T_NATIVE_FLOAT, totalParticles);
        }
        else if (fieldName == "halo_id") {
            dataset_id = createDataset(particles_group_id_, fieldName, H5T_NATIVE_UINT, totalParticles);
        }
        else if (fieldName == "id") {
            dataset_id = createDataset(particles_group_id_, fieldName, H5T_NATIVE_UINT64, totalParticles);
        }
        else {
            throw std::runtime_error("Unknown particle field name: " + fieldName);
        }
        H5Dclose(dataset_id);
    }
        
    hid_t attr_space = H5Screate(H5S_SCALAR);
    h5CheckId(attr_space, "H5Screate(H5S_SCALAR)");
    // TotalParticles (uint64)
    hid_t attr_total = H5Acreate2(file_id_, "TotalParticles", H5T_NATIVE_UINT64,
                                  attr_space, H5P_DEFAULT, H5P_DEFAULT);
    h5CheckId(attr_total, "H5Acreate2(TotalParticles)");
    if (rank_ == 0) {
        std::uint64_t total = static_cast<std::uint64_t>(totalParticles);
        h5Check(H5Awrite(attr_total, H5T_NATIVE_UINT64, &total), "H5Awrite(TotalParticles)");
    }
    H5Aclose(attr_total);
    // NumberOfClusters (uint64)
    hid_t attr_nc = H5Acreate2(file_id_, "NumberOfClusters", H5T_NATIVE_UINT64,
                               attr_space, H5P_DEFAULT, H5P_DEFAULT);
    h5CheckId(attr_nc, "H5Acreate2(NumberOfClusters)");
    if (rank_ == 0) {
        std::uint64_t n = static_cast<std::uint64_t>(clusterInfos.size());
        h5Check(H5Awrite(attr_nc, H5T_NATIVE_UINT64, &n), "H5Awrite(NumberOfClusters)");
    }
    H5Aclose(attr_nc);
    H5Sclose(attr_space);

    MPI_Barrier(comm_);
}


void ClusterHDF5Writer::writeParticles(const std::vector<std::string>& fieldNames,
                                      const HDF5HostData& particles, 
                                      const std::vector<ClusterInfo>& clusterInfos) {

    const bool hasX = std::find(fieldNames.begin(), fieldNames.end(), "x") != fieldNames.end();
    const bool hasY = std::find(fieldNames.begin(), fieldNames.end(), "y") != fieldNames.end();
    const bool hasZ = std::find(fieldNames.begin(), fieldNames.end(), "z") != fieldNames.end();
    const bool hasVx = std::find(fieldNames.begin(), fieldNames.end(), "vx") != fieldNames.end();
    const bool hasVy = std::find(fieldNames.begin(), fieldNames.end(), "vy") != fieldNames.end();
    const bool hasVz = std::find(fieldNames.begin(), fieldNames.end(), "vz") != fieldNames.end();
    const bool hasM = std::find(fieldNames.begin(), fieldNames.end(), "m") != fieldNames.end();
    const bool hasHaloId = std::find(fieldNames.begin(), fieldNames.end(), "halo_id") != fieldNames.end();
    const bool hasId = std::find(fieldNames.begin(), fieldNames.end(), "id") != fieldNames.end();
    
    for (const auto& info : clusterInfos) {
        
        const size_t clusterGlobalOffset = static_cast<size_t>(info.globalOffset) +
                                           static_cast<size_t>(info.haloOffset);
        size_t count = info.localCount;
        size_t localOffset = info.localOffset;
        
        // All ranks must participate in collective writes, even with empty data
        if (hasX) {
            std::vector<double> cluster_x;
            if (count > 0) {
                assert(localOffset + count <= particles.x.size());
                cluster_x = std::vector<double>(particles.x.begin() + localOffset, particles.x.begin() + localOffset + count);
            }
            writeField(particles_group_id_, "x", cluster_x, 0, clusterGlobalOffset);
        }
        
        if (hasY) {
            std::vector<double> cluster_y;
            if (count > 0) {
                assert(localOffset + count <= particles.y.size());
                cluster_y = std::vector<double>(particles.y.begin() + localOffset, particles.y.begin() + localOffset + count);
            }
            writeField(particles_group_id_, "y", cluster_y, 0, clusterGlobalOffset);
        }
        
        if (hasZ) {
            std::vector<double> cluster_z;
            if (count > 0) {
                assert(localOffset + count <= particles.z.size());
                cluster_z = std::vector<double>(particles.z.begin() + localOffset, particles.z.begin() + localOffset + count);
            }
            writeField(particles_group_id_, "z", cluster_z, 0, clusterGlobalOffset);
        }
        
        if (hasVx) {
            std::vector<float> cluster_vx;
            if (count > 0) {
                assert(localOffset + count <= particles.vx.size());
                cluster_vx = std::vector<float>(particles.vx.begin() + localOffset, particles.vx.begin() + localOffset + count);
            }
            writeField(particles_group_id_, "vx", cluster_vx, 0, clusterGlobalOffset);
        }
        
        if (hasVy) {
            std::vector<float> cluster_vy;
            if (count > 0) {
                assert(localOffset + count <= particles.vy.size());
                cluster_vy = std::vector<float>(particles.vy.begin() + localOffset, particles.vy.begin() + localOffset + count);
            }
            writeField(particles_group_id_, "vy", cluster_vy, 0, clusterGlobalOffset);
        }

        if (hasVz) {
            std::vector<float> cluster_vz;
            if (count > 0) {
                assert(localOffset + count <= particles.vz.size());
                cluster_vz = std::vector<float>(particles.vz.begin() + localOffset, particles.vz.begin() + localOffset + count);
            }
            writeField(particles_group_id_, "vz", cluster_vz, 0, clusterGlobalOffset);
        }
        
        if (hasM) {
            std::vector<float> cluster_m;
            if (count > 0) {
                assert(localOffset + count <= particles.m.size());
                cluster_m = std::vector<float>(particles.m.begin() + localOffset, particles.m.begin() + localOffset + count);
            }
            writeField(particles_group_id_, "m", cluster_m, 0, clusterGlobalOffset);
        }
        
        if (hasHaloId) {
            std::vector<unsigned> cluster_halo_id;
            if (count > 0) {
                assert(localOffset + count <= particles.halo_id.size());
                cluster_halo_id = std::vector<unsigned>(particles.halo_id.begin() + localOffset, particles.halo_id.begin() + localOffset + count);
            }
            writeField(particles_group_id_, "halo_id", cluster_halo_id, 0, clusterGlobalOffset);
        }

        if (hasId) {
            std::vector<uint64_t> cluster_id;
            if (count > 0) {
                assert(localOffset + count <= particles.id.size());
                cluster_id = std::vector<uint64_t>(particles.id.begin() + localOffset, particles.id.begin() + localOffset + count);
            }
            writeField(particles_group_id_, "id", cluster_id, 0, clusterGlobalOffset);
        }
    }
}

void ClusterHDF5Writer::writeClusterMetadata(const std::vector<ClusterInfo>& clusterInfos) {

    size_t numClusters = clusterInfos.size();
    
    // Create datasets for cluster metadata
    hid_t cluster_ids_dset = createDataset(clusters_group_id_, "cluster_ids",
                                          H5T_NATIVE_UINT, numClusters);
    hid_t offsets_dset = createDataset(clusters_group_id_, "global_offsets",
                                       H5T_NATIVE_UINT64, numClusters);
    hid_t counts_dset = createDataset(clusters_group_id_, "particle_counts",
                                      H5T_NATIVE_UINT64, numClusters);
    
    if (rank_ == 0) { // Only rank 0 writes metadata
        // Prepare data arrays
        std::vector<unsigned> cluster_ids(numClusters);
        std::vector<std::uint64_t> offsets(numClusters);
        std::vector<std::uint64_t> counts(numClusters);
        
        for (size_t i = 0; i < numClusters; ++i) {
            cluster_ids[i] = clusterInfos[i].clusterId;
            offsets[i] = static_cast<std::uint64_t>(clusterInfos[i].globalOffset);
            counts[i]  = static_cast<std::uint64_t>(clusterInfos[i].globalCount);
        }

        // Write metadata
        H5Dwrite(cluster_ids_dset, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, 
                 H5P_DEFAULT, cluster_ids.data());
        H5Dwrite(offsets_dset, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL,
                 H5P_DEFAULT, offsets.data());
        H5Dwrite(counts_dset, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL,
                 H5P_DEFAULT, counts.data());
        }
    
    // Close datasets immediately after writing
    H5Dclose(cluster_ids_dset);
    H5Dclose(offsets_dset);
    H5Dclose(counts_dset);
    
    // Ensure all ranks synchronize before closing
    MPI_Barrier(comm_);
}

void ClusterHDF5Writer::close() {
    if (particles_group_id_ >= 0) {
        H5Gclose(particles_group_id_);
        particles_group_id_ = H5I_INVALID_HID;
    }
    if (clusters_group_id_ >= 0) {
        H5Gclose(clusters_group_id_);
        clusters_group_id_ = H5I_INVALID_HID;
    }
    if (file_id_ >= 0) {
        // H5Fflush is collective-ish in many parallel builds; make sure all ranks participate.
        MPI_Barrier(comm_);
        H5Fflush(file_id_, H5F_SCOPE_GLOBAL);
        MPI_Barrier(comm_);
        H5Fclose(file_id_);
        file_id_ = H5I_INVALID_HID;
    }
}

// Helper methods implementation
hid_t ClusterHDF5Writer::createGroup(hid_t parent_id, const std::string& name) {
    hid_t group_id = H5Gcreate2(parent_id, name.c_str(), H5P_DEFAULT, 
                               H5P_DEFAULT, H5P_DEFAULT);
    h5CheckId(group_id, ("H5Gcreate2(" + name + ")").c_str());
    return group_id;
}

hid_t ClusterHDF5Writer::createDataset(hid_t group_id, const std::string& name, 
                                      hid_t datatype, size_t total_size) {
    hsize_t dims[1] = {total_size};
    hid_t space_id = H5Screate_simple(1, dims, NULL);
    h5CheckId(space_id, "H5Screate_simple");
    
    hid_t dset_id = H5Dcreate2(group_id, name.c_str(), datatype, space_id,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(space_id);
    h5CheckId(dset_id, ("H5Dcreate2(" + name + ")").c_str());
    return dset_id;
}

template<typename T>
void ClusterHDF5Writer::writeField(hid_t group_id, const std::string& name, 
                                  const std::vector<T>& data, size_t total_size,
                                  size_t global_offset) {
    
    hid_t dset_id = H5Dopen2(group_id, name.c_str(), H5P_DEFAULT);
    h5CheckId(dset_id, ("H5Dopen2(" + name + ")").c_str());
    
    hid_t mem_space = H5Screate(H5S_SCALAR);
    h5CheckId(mem_space, "H5Screate(H5S_SCALAR)");
    if (data.empty()) {
        h5Check(H5Sselect_none(mem_space), "H5Sselect_none(mem_space)");
    } else {
        H5Sclose(mem_space);
        hsize_t local_dims[1] = {static_cast<hsize_t>(data.size())};
        mem_space = H5Screate_simple(1, local_dims, NULL);
        h5CheckId(mem_space, "H5Screate_simple(mem_space)");
    }
    
    // Get file space and select hyperslab
    hid_t file_space = H5Dget_space(dset_id);
    h5CheckId(file_space, "H5Dget_space");
    
    if (data.empty()) {
        // For empty data, select nothing in the file space
        h5Check(H5Sselect_none(file_space), "H5Sselect_none(file_space)");
    } else {
        hsize_t offset[1] = {global_offset};
        hsize_t count[1] = {static_cast<hsize_t>(data.size())};
        h5Check(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, offset, NULL, count, NULL),
                "H5Sselect_hyperslab");
    }
    
    // Set up collective write (all ranks must participate)
    hid_t xfer_plist = H5Pcreate(H5P_DATASET_XFER);
    h5CheckId(xfer_plist, "H5Pcreate(H5P_DATASET_XFER)");
    h5Check(H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE), "H5Pset_dxpl_mpio(COLLECTIVE)");
    
    // Write data (even if empty, rank must participate in collective operation)
    hid_t h5_type = getHDF5Type<T>();
    const void* write_data = nullptr;
    T dummy{};
    write_data = data.empty() ? static_cast<const void*>(&dummy) : static_cast<const void*>(data.data());

    herr_t status = H5Dwrite(dset_id, h5_type, mem_space, file_space, xfer_plist, write_data);
    h5Check(status, ("H5Dwrite(" + name + ")").c_str());
    
    // Cleanup
    H5Pclose(xfer_plist);
    H5Sclose(file_space);
    H5Sclose(mem_space);
    H5Dclose(dset_id);
}

template<typename T>
hid_t ClusterHDF5Writer::getHDF5Type() {
    if constexpr (std::is_same_v<T, float>) return H5T_NATIVE_FLOAT;
    else if constexpr (std::is_same_v<T, double>) return H5T_NATIVE_DOUBLE;
    else if constexpr (std::is_same_v<T, int>) return H5T_NATIVE_INT;
    else if constexpr (std::is_same_v<T, unsigned>) return H5T_NATIVE_UINT;
    else if constexpr (std::is_same_v<T, long>) return H5T_NATIVE_LONG;
    else if constexpr (std::is_same_v<T, unsigned long>) return H5T_NATIVE_ULONG;
    else if constexpr (std::is_same_v<T, uint64_t>) return H5T_NATIVE_UINT64;
    else if constexpr (std::is_same_v<T, int64_t>) return H5T_NATIVE_INT64;
    else {
        static_assert(sizeof(T) == 0, "Unsupported type for HDF5");
        return H5T_NATIVE_DOUBLE;
    }
}

// Explicit template instantiations for helper methods
template void ClusterHDF5Writer::writeField<float>(hid_t, const std::string&, 
                                                  const std::vector<float>&, size_t, size_t);
template void ClusterHDF5Writer::writeField<double>(hid_t, const std::string&, 
                                                   const std::vector<double>&, size_t, size_t);
template void ClusterHDF5Writer::writeField<unsigned>(hid_t, const std::string&, 
                                                     const std::vector<unsigned>&, size_t, size_t);
template void ClusterHDF5Writer::writeField<uint64_t>(hid_t, const std::string&,
                                                     const std::vector<uint64_t>&, size_t, size_t);


} // namespace sphexa