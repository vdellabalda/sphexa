cmake_minimum_required(VERSION 3.22)
set(CMAKE_CXX_STANDARD 20)

project(cluster LANGUAGES CXX CUDA)

set(CSTONE_DIR ${CMAKE_SOURCE_DIR}/domain/include)
set(MAIN_APP_DIR ${CMAKE_SOURCE_DIR}/main/src)
set(CLUSTER_DIR ${CMAKE_SOURCE_DIR}/cluster/include)
set(SPH_DIR ${CMAKE_SOURCE_DIR}/sph/include)

add_subdirectory(include/cluster)
#add_subdirectory(include/halo)

