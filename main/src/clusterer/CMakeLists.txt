add_library(clusterer dark_clust.cpp)
target_include_directories(clusterer PRIVATE ${PROJECT_SOURCE_DIR}/main/src ${CSTONE_DIR} 
        ${SPH_DIR} ${COOLING_DIR} ${CLUSTER_DIR} ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(clusterer PRIVATE ${MPI_CXX_LIBRARIES} util OpenMP::OpenMP_CXX)

if (CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(clusterer_gpu dark_clust.cpp)
    target_compile_definitions(clusterer_gpu PRIVATE USE_CUDA)
    target_include_directories(clusterer_gpu PRIVATE ${PROJECT_SOURCE_DIR}/main/src ${CSTONE_DIR}
            ${SPH_DIR} ${COOLING_DIR} ${CLUSTER_DIR} ${MPI_CXX_INCLUDE_PATH})
    target_link_libraries(clusterer_gpu PRIVATE ${MPI_CXX_LIBRARIES} cstone_gpu sph_gpu cluster_gpu util OpenMP::OpenMP_CXX)
endif ()

if (CMAKE_HIP_COMPILER)
    target_link_libraries(clusterer_gpu PRIVATE hip::host)
    target_compile_definitions(clusterer_gpu PRIVATE THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP)
    set_target_properties(clusterer_gpu PROPERTIES LINKER_LANGUAGE CXX)
endif ()

if (CMAKE_CUDA_COMPILER)
    target_link_libraries(clusterer_gpu PRIVATE CUDA::cudart)
endif ()

# Standalone clustering executable
set(cluster_exename cluster)
add_executable(${cluster_exename} cluster.cpp)
target_include_directories(${cluster_exename} PRIVATE ${SPH_EXA_INCLUDE_DIRS})
target_link_libraries(${cluster_exename} PRIVATE io sim_init propagator clusterer OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES})

install(TARGETS ${cluster_exename} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if (CMAKE_CUDA_COMPILER)
    add_executable(${cluster_exename}-cuda cluster.cpp)
    target_include_directories(${cluster_exename}-cuda PRIVATE ${SPH_EXA_INCLUDE_DIRS})
    target_include_directories(${cluster_exename}-cuda PUBLIC ${CMAKE_BINARY_DIR}/main/src)
    target_compile_definitions(${cluster_exename}-cuda PRIVATE USE_CUDA)
    target_link_libraries(${cluster_exename}-cuda PRIVATE io sim_init_gpu propagator_gpu clusterer_gpu OpenMP::OpenMP_CXX
        ${MPI_CXX_LIBRARIES} CUDA::cudart)
    target_include_directories(${cluster_exename}-cuda PRIVATE ${PROJECT_SOURCE_DIR}/physics/cooling/include/)
    install(TARGETS ${cluster_exename}-cuda RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

elseif (CMAKE_HIP_COMPILER)
    add_executable(${cluster_exename}-hip cluster.cpp)
    target_include_directories(${cluster_exename}-hip PRIVATE ${SPH_EXA_INCLUDE_DIRS})
    target_compile_definitions(${cluster_exename}-hip PRIVATE USE_CUDA THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP)
    target_link_libraries(${cluster_exename}-hip PRIVATE io sim_init_gpu propagator_gpu clusterer_gpu OpenMP::OpenMP_CXX
        ${MPI_CXX_LIBRARIES} hip::host)
    target_include_directories(${cluster_exename}-hip PRIVATE ${PROJECT_SOURCE_DIR}/physics/cooling/include/)
    set_target_properties(${cluster_exename}-hip PROPERTIES LINKER_LANGUAGE CXX)
    install(TARGETS ${cluster_exename}-hip RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif ()


