cmake_minimum_required(VERSION 3.9...3.28)

project(MasterThesis VERSION 1.0
DESCRIPTION "Compressed Matrix Multiplication"
LANGUAGES CXX)
set (CMAKE_CXX_STANDARD 17)

if(DEFINED CPP_ENV)
  message("CppEnv detected")
  link_directories(CPP_ENV)
endif(DEFINED CPP_ENV)

# add_definitions(-DUSE_MKL)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Disable the Google Benchmark requirement on Google Test
set(BENCHMARK_ENABLE_TESTING NO)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${M_ARCH}")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

set(BLAS_PROVIDED ON)

option(USE_ACCELERATE "USE_ACCELERATE" OFF)
option(USE_MKL "USE_MKL" OFF)
option(USE_AOCL "USE_AOCL" OFF)
option(USE_OPENBLAS "USE_OPENBLAS" OFF)

if(USE_ACCELERATE)
  message("Apple Accelerate selected")
  set(CMAKE_CXX_FLAGS_RELEASE "-D USE_ACCELERATE ${CMAKE_CXX_FLAGS_RELEASE}")
  set(BLA_VENDER Apple)
elseif(USE_MKL)
  message("Intel MKL selected")
  set(CMAKE_CXX_FLAGS_RELEASE "-D USE_MKL ${CMAKE_CXX_FLAGS_RELEASE}")
  set(BLA_VENDER Intel10_64lp_seq)
elseif(USE_AOCL)
  message("AMD AOCL selected")
  set(CMAKE_CXX_FLAGS_RELEASE "-D USE_AOCL ${CMAKE_CXX_FLAGS_RELEASE}")
  set(BLA_VENDER AOCL_mt)
  if(DEFINED ENV{AOCL_ROOT})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I$ENV{AOCL_ROOT}/include -L$ENV{AOCL_ROOT}/lib -lfftw3 -lblis-mt")
  elseif(EXISTS "/opt/AMD/aocl/aocl-linux-gcc-4.2.0/gcc")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/opt/AMD/aocl/aocl-linux-gcc-4.2.0/gcc/include -L/opt/AMD/aocl/aocl-linux-gcc-4.2.0/gcc/lib -lfftw3 -lblis-mt")
  endif(DEFINED ENV{AOCL_ROOT})
elseif(USE_OPENBLAS)
  message("OpenBLAS selected")
  set(CMAKE_CXX_FLAGS_RELEASE "-D USE_OPENBLAS ${CMAKE_CXX_FLAGS_RELEASE}")
  set(BLA_VENDER OpenBLAS)
else()
  set(BLAS_PROVIDED OFF)
endif()

enable_testing()

if(BLAS_PROVIDED)
  find_package(BLAS REQUIRED)
  include_directories(${BLAS_INCLUDE_DIRS})
endif(BLAS_PROVIDED)

# Imports libraries
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Catch2 3 REQUIRED)
find_package(benchmark REQUIRED)
find_package(OpenMP REQUIRED)

if(NOT USE_AOCL)
  find_package(PkgConfig REQUIRED)
  pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)
  include_directories(PkgConfig::FFTW)
  set(fftwVar PkgConfig::FFTW)
else()
  set(fftwVar fftw3)
endif()

############    temp    ############

add_executable(temp "benchmark/temp.cpp")

target_link_libraries(temp Eigen3::Eigen)
target_link_libraries(temp benchmark::benchmark)

if(OpenMP_FOUND)
  message("temp: OpenMP found")
  target_link_libraries(temp OpenMP::OpenMP_CXX)
endif(OpenMP_FOUND)

if(BLAS_FOUND)
  message("temp: BLAS found")
  target_link_libraries(temp ${BLAS_LIBRARIES})
endif(BLAS_FOUND)

############ Benchmarks ############

# file(GLOB_RECURSE BENCHMARKS benchmark/benchmark_compressed_mul.cpp benchmark/benchmark_compressed_mul.hpp src/*.cpp src/*.hpp include/*.hpp)
# FOREACH(item ${BENCHMARKS})
#   IF(${item} MATCHES "src/main.cpp")
#     LIST(REMOVE_ITEM BENCHMARKS ${item})
#   ENDIF(${item} MATCHES "src/main.cpp")
# ENDFOREACH(item)
# add_executable(benchmarks ${BENCHMARKS})

add_executable(benchmarks 
        benchmark/benchmark_compressed_mul.cpp 
        benchmark/benchmark_compressed_mul.hpp
        src/fft.cpp 
        src/fft.hpp
        src/hashing.hpp
        src/benchmark_timer.cpp                 
        src/benchmark_timer.hpp                 
        src/utils.cpp                 
        src/utils.hpp                 
        include/rapidcsv.h)

target_link_libraries(benchmarks Eigen3::Eigen)
target_link_libraries(benchmarks ${fftwVar})
target_link_libraries(benchmarks benchmark::benchmark)

if(OpenMP_FOUND)
  message("benchmarks: OpenMP found")
  target_link_libraries(benchmarks OpenMP::OpenMP_CXX)
endif(OpenMP_FOUND)

if(BLAS_FOUND)
  message("benchmarks: BLAS found")
  target_link_libraries(benchmarks ${BLAS_LIBRARIES})
endif(BLAS_FOUND)

############ Input generator ############

add_executable(input_generator 
        benchmark/input_generator.cpp 
        src/utils.cpp 
        src/utils.hpp                 
        include/rapidcsv.h)

target_link_libraries(input_generator Eigen3::Eigen)
target_link_libraries(input_generator ${fftwVar})

if(OpenMP_FOUND)
message("input_generator: OpenMP found")
  target_link_libraries(input_generator OpenMP::OpenMP_CXX)
endif(OpenMP_FOUND)

if(BLAS_FOUND)
  message("input_generator: BLAS found")
  target_link_libraries(input_generator ${BLAS_LIBRARIES})
endif(BLAS_FOUND)



############ Main program ############

file(GLOB_RECURSE SOURCES src/*.cpp src/*.hpp include/*.hpp)
FOREACH(item ${SOURCES})
  IF(${item} MATCHES "src/variance.cpp")
    LIST(REMOVE_ITEM SOURCES ${item})
  ENDIF(${item} MATCHES "src/variance.cpp")
ENDFOREACH(item)
add_executable(app ${SOURCES})

target_link_libraries(app Eigen3::Eigen)
target_link_libraries(app ${fftwVar})

if(OpenMP_FOUND)
  message("app: OpenMP found")
  target_link_libraries(app OpenMP::OpenMP_CXX)
endif(OpenMP_FOUND)

if(BLAS_FOUND)
  message("app: BLAS found")
  target_link_libraries(app ${BLAS_LIBRARIES})
endif(BLAS_FOUND)


############ Testing ############

# Removes main.cpp from the test executable
file(GLOB_RECURSE TESTS test/*.cpp test/*.hpp src/*.cpp src/*.hpp include/*.hpp)
FOREACH(item ${TESTS})
  IF(${item} MATCHES "src/main.cpp")
    LIST(REMOVE_ITEM TESTS ${item})
  ENDIF(${item} MATCHES "src/main.cpp")
ENDFOREACH(item)

add_executable(tests ${TESTS})
# These tests can use the Catch2-provided main
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(tests PUBLIC Eigen3::Eigen)
target_link_libraries(tests PUBLIC ${fftwVar})

if(OpenMP_FOUND)
  target_link_libraries(tests PUBLIC OpenMP::OpenMP_CXX)
endif(OpenMP_FOUND)

if(BLAS_FOUND)
  target_link_libraries(tests PUBLIC ${BLAS_LIBRARIES})
endif(BLAS_FOUND)


############ BLAS CONFIG ############

if(BLAS_PROVIDED)
  if(USE_MKL)
    target_compile_definitions(app PUBLIC EIGEN_USE_MKL_ALL)
    target_compile_definitions(tests PUBLIC EIGEN_USE_MKL_ALL)
    target_compile_definitions(benchmarks PUBLIC EIGEN_USE_MKL_ALL)
    target_compile_definitions(input_generator PUBLIC EIGEN_USE_MKL_ALL)
    target_compile_definitions(temp PUBLIC EIGEN_USE_MKL_ALL)
  elseif (NOT USE_MKL)
    target_compile_definitions(app PUBLIC EIGEN_USE_BLAS)
    target_compile_definitions(tests PUBLIC EIGEN_USE_BLAS)
    target_compile_definitions(benchmarks PUBLIC EIGEN_USE_BLAS)
    target_compile_definitions(input_generator PUBLIC EIGEN_USE_BLAS)
    target_compile_definitions(temp PUBLIC EIGEN_USE_BLAS)
  endif()
endif(BLAS_PROVIDED)


# Uncomment this to disable eigen parallelism
# add_compile_definitions(EIGEN_DONT_PARALLELIZE)

include(CTest)
include(Catch)
catch_discover_tests(tests)