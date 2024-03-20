#include "benchmark_compressed_mul.hpp"

#include "../include/cxxopts.hpp"
#include "../src/compressed_mul.hpp"
#include "../src/hashing.hpp"

#include <benchmark/benchmark.h>

#include <iostream>
#include <random>



static void BM_eigen(benchmark::State& state, const MatrixRXd& m1, const MatrixRXd& m2) {
    MatrixRXd result;
    for (auto _ : state) {
        result = m1 * m2;
    };
}

int main(int argc, char** argv) {
    cxxopts::Options options("parameters", "Parameters to test with");

    options.allow_unrecognised_options();

    // clang-format off
    options.add_options()
        ("n,size", "Size", cxxopts::value<int>())
        ("b,bb", "b", cxxopts::value<std::vector<int64_t>>())
        ("d,dd", "d", cxxopts::value<std::vector<int64_t>>())
        ("p,density", "Density", cxxopts::value<double>())
        ("s,seed", "The random seed", cxxopts::value<unsigned int>())
        ("h,debug", "Enable debugging", cxxopts::value<bool>())    
        ("full", "Enable fully random", cxxopts::value<bool>())
        ("mul", "Enable multiply-shift", cxxopts::value<bool>())
        ("tab", "Enable tabulation", cxxopts::value<bool>())
        ("thread_count", "Number of threads to use", cxxopts::value<int>())
        // ("bench", "What benchmarks to run", cxxopts::value<std::vector<std::string>>())
    ;
    // clang-format on

    options.parse_positional({"n", "density", "seed"});
    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << "TODO: some help" << std::endl;
        exit(0);
    }

    int thread_count = result["thread_count"].as<int>();
    omp_set_num_threads(thread_count);

    int n = result["n"].as<int>();
    std::vector<int64_t> bs = result["bb"].as<std::vector<int64_t>>();
    std::vector<int64_t> ds = result["dd"].as<std::vector<int64_t>>();

    double density;
    if (result.count("density")) {
        density = result["density"].as<double>();
    } else {
        density = 5.0 / n;  // Might need to be adjusted
    }

    bool full = result["full"].as<bool>();
    bool mul = result["mul"].as<bool>();
    bool tab = result["tab"].as<bool>();

    unsigned long seed;
    if (result.count("seed")) {
        seed = result["seed"].as<unsigned int>();
    } else {
        seed = std::random_device{}();
    }
    std::mt19937_64 rng(seed);


    // Get the input matrices from the text stream
    MatrixRXd m1(n, n), m2(n, n);
    std::cin.read(reinterpret_cast<char*>(m1.data()), n * n * sizeof(double));
    std::cin.read(reinterpret_cast<char*>(m2.data()), n * n * sizeof(double));

    std::cout << "----- Settings -----" << std::endl;
    std::cout << "n = " << n << std::endl;
    std::cout << "density = " << density << std::endl;
#ifdef USE_MKL
    std::cout << "Backend = MKL" << std::endl;
#elif USE_ACCELERATE
    std::cout << "Backend = ACCELERATE" << std::endl;
#else
    std::cout << "Backend = FFTW" << std::endl;
#endif
    std::cout << "Seed = " << seed << std::endl;
    std::cout << "----- Settings -----" << std::endl;

    // Register the parameterized benchmark with the input values

    benchmark::RegisterBenchmark("Eigen", BM_eigen, m1, m2)
        ->Iterations(100)
        ->Repetitions(10)
        ->DisplayAggregatesOnly(true);
    // benchmark::RegisterBenchmark("BM_block", BM_block) -> Args({b, d});

    if (full) {
        benchmark::RegisterBenchmark("compressed_product_par fully_random", BM_compressed_product_random_par<int>, m1, m2, seed)
            ->ArgsProduct({bs, ds})
            ->Iterations(100)
            ->Repetitions(10)
            ->DisplayAggregatesOnly(true);
        benchmark::RegisterBenchmark("decompress_matrix_par fully_random", BM_decompress_matrix_random_par<int>, m1, m2, seed)
            ->ArgsProduct({bs, ds})
            ->Iterations(100)
            ->Repetitions(10)
            ->DisplayAggregatesOnly(true);
    }

    if (mul) {
        benchmark::RegisterBenchmark("compressed_product_par multiply-shift", BM_compressed_product_multiply_shift_par<uint32_t, uint16_t>, m1, m2, seed)
            ->ArgsProduct({bs, ds})
            ->Iterations(100)
            ->Repetitions(10)
            ->DisplayAggregatesOnly(true);
        benchmark::RegisterBenchmark("decompress_matrix_par multiply-shift", BM_decompress_matrix_multiply_shift_par<uint32_t, uint16_t>, m1, m2, seed)
            ->ArgsProduct({bs, ds})
            ->Iterations(100)
            ->Repetitions(10)
            ->DisplayAggregatesOnly(true);
    }

    if (tab) {
        benchmark::RegisterBenchmark("compressed_product_par tabulation", BM_compressed_product_tabulation_par<uint32_t, uint32_t, 8>, m1, m2, seed)
            ->ArgsProduct({bs, ds})
            ->Iterations(100)
            ->Repetitions(10)
            ->DisplayAggregatesOnly(true);
        benchmark::RegisterBenchmark("decompress_matrix_par tabulation", BM_decompress_matrix_tabulation_par<uint32_t, uint32_t, 8>, m1, m2, seed)
            ->ArgsProduct({bs, ds})
            ->Iterations(100)
            ->Repetitions(10)
            ->DisplayAggregatesOnly(true);
    }

    // Initialize Google Benchmark and run specified benchmarks
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}
