cmake_minimum_required( VERSION 3.15 )

project(puppet)

# ------------ Find the libraries we need --------------------------

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set( CMAKE_PREFIX_PATH
    ${CMAKE_SOURCE_DIR}/libs/
    ${CMAKE_SOURCE_DIR}/libs/${CMAKE_BUILD_TYPE}
    ${CMAKE_PREFIX_PATH})

find_package(LAPACK COMPONENTS MKL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(gflags REQUIRED)
find_package(Ceres REQUIRED)

# ------------------- Settings --------------------------------------

# use C++11
set (CMAKE_CXX_STANDARD 11)
  
# -------------------- Walk the subdirectories --------------------

add_subdirectory( src )
