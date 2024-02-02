cmake_minimum_required(VERSION 3.5...3.28)

include(FetchContent)

project(MasterThesis VERSION 1.0
                     DESCRIPTION "Compressed Matrix Multiplication"
                     LANGUAGES CXX)


# FetchContent_Declare(fftw URL https://www.fftw.org/fftw-3.3.10.tar.gz)
# FetchContent_MakeAvailable(fftw)

add_executable(app src/main.cpp)