# CPRA
# Fangzhou Ai, UCSD @ Prof. Vitaliy Lomakin's group

cmake_minimum_required(VERSION 3.18)

project(CPRA CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# CUDA version
option(HAS_CUDA CACHE ON)
# Intel MKL
option(HAS_MKL CACHE ON)
# Build test
option(BUILD_TEST CACHE OFF)
# Set output path
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Build GLOG
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/glog)
include_directories(${CMAKE_SOURCE_DIR}/third_party/glog)                      # glog header


# CUDA_enabled device
if(HAS_CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_definitions(-DHAS_CUDA)
endif()

# Intel MKL
if(HAS_MKL)
    find_package(MKL CONFIG REQUIRED)
    add_definitions(-DHAS_MKL)
endif()

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    add_definitions(-DHAS_OMP)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
add_subdirectory(${CMAKE_SOURCE_DIR}/src)

if(BUILD_TEST)
    enable_testing()
    add_subdirectory(${CMAKE_SOURCE_DIR}/test)
endif()