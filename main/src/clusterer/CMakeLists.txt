add_library(clusterer dark_clust.cpp)
target_include_directories(clusterer PRIVATE ${PROJECT_SOURCE_DIR}/main/src ${CSTONE_DIR}
        ${SPH_DIR} ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(clusterer PRIVATE ${MPI_CXX_LIBRARIES} util OpenMP::OpenMP_CXX)

if (CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(clusterer_gpu dark_clust.cpp)
    target_compile_definitions(clusterer_gpu PRIVATE USE_CUDA)
    target_include_directories(clusterer_gpu PRIVATE ${PROJECT_SOURCE_DIR}/main/src ${CSTONE_DIR}
            ${SPH_DIR} ${MPI_CXX_INCLUDE_PATH})
    target_link_libraries(clusterer_gpu PRIVATE ${MPI_CXX_LIBRARIES} cstone_gpu sph_gpu util OpenMP::OpenMP_CXX)
endif ()

if (CMAKE_HIP_COMPILER)
    target_link_libraries(clusterer_gpu PRIVATE hip::host)
    target_compile_definitions(clusterer_gpu PRIVATE THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP)
    set_target_properties(clusterer_gpu PROPERTIES LINKER_LANGUAGE CXX)
endif ()

if (CMAKE_CUDA_COMPILER)
    target_link_libraries(clusterer_gpu PRIVATE CUDA::cudart)
endif ()



