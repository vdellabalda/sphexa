cmake_minimum_required(VERSION 3.22)

project(clusterer CXX)

add_subdirectory(include)
if (BUILD_TESTING)
  add_subdirectory(test)
endif ()
