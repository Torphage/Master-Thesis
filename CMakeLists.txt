cmake_minimum_required(VERSION 3.9...3.28)

project(MasterThesis VERSION 1.0
DESCRIPTION "Compressed Matrix Multiplication"
LANGUAGES CXX)
set (CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()


set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wunused-parameter")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

set(BLAS_PROVIDED ON)

option(USE_ACCELERATE "USE_ACCELERATE" OFF)
option(USE_MKL "USE_MKL" OFF)
option(USE_AOCL "USE_AOCL" OFF)
option(USE_OPENBLAS "USE_OPENBLAS" OFF)

if(USE_ACCELERATE)
  message("Apple Accelerate selected")
  set(BLA_VENDER Apple)
elseif(USE_MKL)
  set(BLA_VENDER Intel10_64lp_seq) # Flag EIGEN_USE_MKL_ALL ?
elseif(USE_AOCL)
  set(BLA_VENDER AOCL)
elseif(USE_OPENBLAS)
  set(BLA_VENDER OpenBLAS)
else()
  set(BLAS_PROVIDED OFF)
endif() 


enable_testing()

if(BLAS_PROVIDED)
  find_package(BLAS REQUIRED)
endif(BLAS_PROVIDED)

# Imports libraries
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Catch2 3 REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)
include_directories(PkgConfig::FFTW)


find_package(OpenMP REQUIRED)


# Main program
file(GLOB_RECURSE SOURCES src/*.cpp src/*.hpp src/*.tpp)

add_executable(app ${SOURCES})

target_link_libraries(app Eigen3::Eigen)
target_link_libraries(app PkgConfig::FFTW)

if(OpenMP_FOUND)
  target_link_libraries(app OpenMP::OpenMP_CXX)
endif(OpenMP_FOUND)

# New stuff, testing
if(BLAS_FOUND)
  message("BLAS found")
  include_directories(${BLAS_INCLUDE_DIRS})
  target_link_libraries(app ${BLAS_LIBRARIES})
endif(BLAS_FOUND)



# Removes main.cpp from the test executable
file(GLOB_RECURSE TESTS test/*.cpp test/*.hpp test/*.tpp src/*.cpp src/*.hpp src/*.tpp)
FOREACH(item ${TESTS})
  IF(${item} MATCHES "src/main.cpp")
    LIST(REMOVE_ITEM TESTS ${item})
  ENDIF(${item} MATCHES "src/main.cpp")
ENDFOREACH(item)

add_executable(tests ${TESTS})
# These tests can use the Catch2-provided main
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(tests PUBLIC Eigen3::Eigen)
target_link_libraries(tests PUBLIC PkgConfig::FFTW)

if(OpenMP_FOUND)
  target_link_libraries(tests PUBLIC OpenMP::OpenMP_CXX)
endif(OpenMP_FOUND)

# New stuff, testing
if(BLAS_FOUND)
  target_link_libraries(tests PUBLIC ${BLAS_LIBRARIES})
endif(BLAS_FOUND)

if(BLAS_PROVIDED)
  target_compile_definitions(app PUBLIC EIGEN_USE_BLAS)
  target_compile_definitions(tests PUBLIC EIGEN_USE_BLAS)
endif(BLAS_PROVIDED)



# Uncomment this to disable eigen parallelism 
# add_compile_definitions(EIGEN_DONT_PARALLELIZE)

include(CTest)
include(Catch)
catch_discover_tests(tests)