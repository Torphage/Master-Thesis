cmake_minimum_required(VERSION 3.9...3.28)

project(MasterThesis VERSION 1.0
DESCRIPTION "Compressed Matrix Multiplication"
LANGUAGES CXX)
set (CMAKE_CXX_STANDARD 17)

enable_testing()



# Imports libraries
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Catch2 3 REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)
include_directories(PkgConfig::FFTW)


find_package(OpenMP REQUIRED)


# Main program
file(GLOB_RECURSE SOURCES src/*.cpp src/*.hpp)

add_executable(app ${SOURCES})
target_link_libraries(app OpenMP::OpenMP_CXX)
target_link_libraries(app Eigen3::Eigen)
target_link_libraries(app PkgConfig::FFTW)


# Testing
file(GLOB_RECURSE TESTS test/*.cpp test/*.hpp src/*.cpp src/*.hpp)
FOREACH(item ${TESTS})
  IF(${item} MATCHES "src/main.cpp")
    LIST(REMOVE_ITEM TESTS ${item})
  ENDIF(${item} MATCHES "src/main.cpp")
ENDFOREACH(item)

add_executable(tests ${TESTS})
# These tests can use the Catch2-provided main
target_link_libraries(tests PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(tests PUBLIC Eigen3::Eigen)
target_link_libraries(tests PUBLIC PkgConfig::FFTW)
# target_link_libraries(tests PUBLIC OpenMP::OpenMP_CXX)


# Ensure that changes in test source files trigger a rebuild
# add_custom_target(tests_rebuild DEPENDS ${TESTS})


include(CTest)
include(Catch)
catch_discover_tests(tests)