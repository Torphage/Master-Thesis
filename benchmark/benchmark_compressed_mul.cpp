#include <benchmark/benchmark.h>

#include <iostream>
#include <random>

#include "../include/cxxopts.hpp"
#include "../src/compressed_mul.hpp"

static void BM_compressed_product_fully_random(benchmark::State& state, const MatrixRXd& m1, const MatrixRXd& m2, std::mt19937_64 rng) {
    int n = m1.rows();
    int b = state.range(0);
    int d = state.range(1);

    Hashes<Eigen::MatrixXi> hashes = fully_random_constructor(n, b, d, rng);

    MatrixRXd result;
    for (auto _ : state) {
        benchmark::DoNotOptimize(result = compressed_product_par(m1, m2, b, d, fully_random_hash(), hashes));
    };
}

static void BM_decompress_matrix_fully_random(benchmark::State& state, const MatrixRXd& m1, const MatrixRXd& m2, std::mt19937_64 rng) {
    int n = m1.rows();
    int b = state.range(0);
    int d = state.range(1);

    Hashes<Eigen::MatrixXi> hashes = fully_random_constructor(n, b, d, rng);
    MatrixRXd compressed = compressed_product_par(m1, m2, b, d, fully_random_hash(), hashes);

    MatrixRXd result;
    for (auto _ : state) {
        result = decompress_matrix_par(compressed, n, b, d, fully_random_hash(), hashes);
    };
}

static void BM_compressed_product_multiply_shift(benchmark::State& state, const MatrixRXd& m1, const MatrixRXd& m2, std::mt19937_64 rng) {
    int b = state.range(0);
    int d = state.range(1);

    Hashes<MatrixXui> hashes = multiply_shift_constructor(d, rng);

    for (auto _ : state) {
        compressed_product_par(m1, m2, b, d, multiply_shift_hash(), hashes, b);
    };
}

static void BM_decompress_matrix_multiply_shift(benchmark::State& state, const MatrixRXd& m1, const MatrixRXd& m2, std::mt19937_64 rng) {
    int n = m1.rows();
    int b = state.range(0);
    int d = state.range(1);

    Hashes<MatrixXui> hashes = multiply_shift_constructor(d, rng);
    MatrixRXd compressed = compressed_product_par(m1, m2, b, d, multiply_shift_hash(), hashes, b);

    MatrixRXd result;
    for (auto _ : state) {
        result = decompress_matrix_par(compressed, n, b, d, multiply_shift_hash(), hashes, b);
    };
}

static void BM_compressed_product_tabulation(benchmark::State& state, const MatrixRXd& m1, const MatrixRXd& m2, std::mt19937_64 rng) {
    int b = state.range(0);
    int d = state.range(1);
    int p = state.range(2);
    int q = state.range(3);
    int r = state.range(4);

    Hashes<std::vector<MatrixXui>> hashes = tabulation_constructor(p, q, r, d, rng);

    int t = ceil(p / r);

    for (auto _ : state) {
        compressed_product_par(m1, m2, b, d, tabulation_hash(), hashes, b, r, t);
    };
}

static void BM_decompress_matrix_tabulation(benchmark::State& state, const MatrixRXd& m1, const MatrixRXd& m2, std::mt19937_64 rng) {
    int n = m1.rows();
    int b = state.range(0);
    int d = state.range(1);
    int p = state.range(2);
    int q = state.range(3);
    int r = state.range(4);

    int t = ceil(p / r);

    Hashes<std::vector<MatrixXui>> hashes = tabulation_constructor(p, q, r, d, rng);
    MatrixRXd compressed = compressed_product_par(m1, m2, b, d, tabulation_hash(), hashes, b, r, t);

    MatrixRXd result;
    for (auto _ : state) {
        result = decompress_matrix_par(compressed, n, b, d, tabulation_hash(), hashes, b, r, t);
    };
}

static void BM_eigen(benchmark::State& state, const MatrixRXd& m1, const MatrixRXd& m2) {
    MatrixRXd result;
    for (auto _ : state) {
        result = m1 * m2;
    };
}

int main(int argc, char** argv) {
    cxxopts::Options options("parameters", "Parameters to test with");

    // clang-format off
    options.add_options()
        ("s,seed", "The random seed", cxxopts::value<unsigned int>()->default_value(std::to_string(std::random_device{}())))
        ("n,size", "Size", cxxopts::value<int>()->default_value("0"))
        ("b,bb", "b", cxxopts::value<int>()->default_value("0"))
        ("d,dd", "d", cxxopts::value<int>()->default_value("0"))
        ("p,density", "Density", cxxopts::value<double>()->default_value("1.0"))
        ("h,debug", "Enable debugging", cxxopts::value<bool>()->default_value("false"))    
        ("o,benchmark_out", "Enable debugging", cxxopts::value<std::string>()->default_value("benchmark.json"))
    ;
    // clang-format on
    auto result = options.parse(argc, argv);

    unsigned int seed = result["seed"].as<unsigned int>();
    std::mt19937_64 rng(seed);
    int n = result["size"].as<int>();
    int b = result["b"].as<int>();  // Not used currently
    int d = result["d"].as<int>();  // Not used currently
    double density = result["density"].as<double>();

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

    benchmark::RegisterBenchmark("Eigen", BM_eigen, m1, m2);

    benchmark::RegisterBenchmark("compressed_product fully_random", BM_compressed_product_fully_random, m1, m2, rng)
        ->ArgsProduct({{200, 2000}, {1, 20, 40}});
    benchmark::RegisterBenchmark("compressed_product multiply_shift", BM_compressed_product_multiply_shift, m1, m2, rng)
        ->ArgsProduct({{200, 2000}, {1, 20, 40}});
    benchmark::RegisterBenchmark("compressed_product tabulation", BM_compressed_product_tabulation, m1, m2, rng)
        ->ArgsProduct({{200, 2000}, {1, 20, 40}, {32}, {32}, {8}});
    benchmark::RegisterBenchmark("decompress_matrix fully_random", BM_decompress_matrix_fully_random, m1, m2, rng)
        ->ArgsProduct({{200, 2000}, {1, 20, 40}});
    benchmark::RegisterBenchmark("decompress_matrix multiply_shift", BM_decompress_matrix_multiply_shift, m1, m2, rng)
        ->ArgsProduct({{200, 2000}, {1, 20, 40}});
    benchmark::RegisterBenchmark("decompress_matrix tabulation", BM_decompress_matrix_tabulation, m1, m2, rng)
        ->ArgsProduct({{200, 2000}, {1, 20, 40}, {32}, {32}, {8}});

    // Initialize Google Benchmark and run specified benchmarks
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}
