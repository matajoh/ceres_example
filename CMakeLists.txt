cmake_minimum_required( VERSION 3.15 )

include(FetchContent)

project(ceres_example)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# ------------ Fetch the dependencies ------------------------------

FetchContent_Declare(gflags
  GIT_REPOSITORY https://github.com/gflags/gflags/
  GIT_TAG        v2.2.2
)

FetchContent_GetProperties(gflags)
if (NOT gflags_POPULATED)
    FetchContent_Populate(gflags)
endif ()

set(gflags_DIR ${gflags_BINARY_DIR})
add_subdirectory(${gflags_SOURCE_DIR})

find_package(LAPACK COMPONENTS MKL REQUIRED)

FetchContent_Declare(eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen
  GIT_TAG        9623c0c4b9f40e74ba6ba551f86831616bdd488c
)
FetchContent_MakeAvailable(eigen)

FetchContent_Declare(
    ceres
    GIT_REPOSITORY https://github.com/Ceres-solver/ceres-solver
    GIT_TAG 2.0.0
)

set(PROVIDE_UNINSTALL_TARGET OFF)
set(MINIGLOG ON)
FetchContent_MakeAvailable(ceres)

# ------------------- Settings --------------------------------------

# use C++11
set (CMAKE_CXX_STANDARD 11)
  
# -------------------- Walk the subdirectories --------------------

add_executable( ceres_example ceres_example.cpp )
target_link_libraries(ceres_example PUBLIC Ceres::ceres Eigen3::Eigen gflags )
