cmake_minimum_required(VERSION 3.5...3.28)

include(FetchContent)

project(MasterThesis VERSION 1.0
                     DESCRIPTION "Compressed Matrix Multiplication"
                     LANGUAGES CXX)
set (CMAKE_CXX_STANDARD 17)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)
include_directories(PkgConfig::FFTW)


file(GLOB_RECURSE SOURCES src/*.cpp src/*.h)

add_executable(app ${SOURCES})
target_link_libraries(app Eigen3::Eigen)
target_link_libraries(app PkgConfig::FFTW)
