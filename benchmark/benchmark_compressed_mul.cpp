#include "benchmark_compressed_mul.hpp"

#include "../include/rapidcsv.h"
#include "../src/compressed_mul.hpp"
#include "../src/hashing.hpp"

#include <benchmark/benchmark.h>

#include <iostream>
#include <random>

static void eigen(MatrixRXd &m1, MatrixRXd &m2, benchmark_timer::pre_run_info run_info) {
    benchmark_json::benchmarkinfo info = benchmark_timer::benchmark(run_info, [=]() { return (m1.matrix() * m2.matrix()).eval(); });

    benchmark_timer::print_benchmark("Eigen", 0, 0, 0, run_info, info);
}

static void eigen_matmul(MatrixRXd &c, MatrixRXd &m1, MatrixRXd &m2, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                c(i, j) += m1(i, k) * m2(k, j);
            }
        }
    }
}

static void BM_eigen_matmul(int n, int b, int d, MatrixRXd &m1, MatrixRXd &m2, benchmark_timer::pre_run_info run_info) {
    MatrixRXd c = MatrixRXd::Zero(n, n);
    benchmark_json::benchmarkinfo info = benchmark_timer::benchmark(run_info, eigen_matmul, c, m1, m2, n);

    benchmark_timer::print_benchmark("Matmul (Eigen)", n, b, d, run_info, info);
}

static void eigen_matmul_par(MatrixRXd &c, MatrixRXd &m1, MatrixRXd &m2, int n) {
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                c(i, j) += m1(i, k) * m2(k, j);
            }
        }
    }
}

static void BM_eigen_matmul_par(int n, int b, int d, MatrixRXd &m1, MatrixRXd &m2, benchmark_timer::pre_run_info run_info) {
    MatrixRXd c = MatrixRXd::Zero(n, n);
    benchmark_json::benchmarkinfo info = benchmark_timer::benchmark(run_info, eigen_matmul_par, c, m1, m2, n);

    benchmark_timer::print_benchmark("Matmul par (Eigen)", n, b, d, run_info, info);
}

static void matmul(std::vector<std::vector<double>> &c, std::vector<std::vector<double>> &m1, std::vector<std::vector<double>> &m2, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                c[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }
}

static void BM_matmul(int n, int b, int d, MatrixRXd &m1, MatrixRXd &m2, benchmark_timer::pre_run_info run_info) {
    std::vector<std::vector<double>> m1v(n, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> m2v(n, std::vector<double>(n, 0.0));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            m1v[i][j] = m1(i, j);
            m2v[i][j] = m2(i, j);
        }
    }
    std::vector<std::vector<double>> c(n, std::vector<double>(n, 0.0));
    benchmark_json::benchmarkinfo info = benchmark_timer::benchmark(run_info, matmul, c, m1v, m2v, n);

    benchmark_timer::print_benchmark("Matmul", n, b, d, run_info, info);
}

static void matmul_par(std::vector<std::vector<double>> &c, std::vector<std::vector<double>> &m1, std::vector<std::vector<double>> &m2, int n) {
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                c[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }
}

static void BM_matmul_par(int n, int b, int d, MatrixRXd &m1, MatrixRXd &m2, benchmark_timer::pre_run_info run_info) {
    std::vector<std::vector<double>> m1v(n, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> m2v(n, std::vector<double>(n, 0.0));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            m1v[i][j] = m1(i, j);
            m2v[i][j] = m2(i, j);
        }
    }
    std::vector<std::vector<double>> c(n, std::vector<double>(n, 0.0));
    benchmark_json::benchmarkinfo info = benchmark_timer::benchmark(run_info, matmul_par, c, m1v, m2v, n);

    benchmark_timer::print_benchmark("Matmul par", n, b, d, run_info, info);
}

static void eigen_array_product(MatrixRXd &c, MatrixRXd &m1, MatrixRXd &m2) {
    c = m1 * m2;
}

static void BM_eigen_array_product(int n, MatrixRXd &m1, MatrixRXd &m2, benchmark_timer::pre_run_info run_info) {
    MatrixRXd c(n, n);
    benchmark_json::benchmarkinfo info = benchmark_timer::benchmark(run_info, eigen_array_product, c, m1, m2);

    benchmark_timer::print_benchmark("Eigen Array Coeffwise product", n, 0, 0, run_info, info);
}

static void eigen_cwise_product(MatrixRXd &c, MatrixRXd &m1, MatrixRXd &m2) {
    c = m1.cwiseProduct(m2);
}

static void BM_eigen_cwise_product(int n, MatrixRXd &m1, MatrixRXd &m2, benchmark_timer::pre_run_info run_info) {
    MatrixRXd c(n, n);
    benchmark_json::benchmarkinfo info = benchmark_timer::benchmark(run_info, eigen_cwise_product, c, m1, m2);

    benchmark_timer::print_benchmark("Eigen Cwise product", n, 0, 0, run_info, info);
}

static void cwise_product(MatrixRXd &c, int n, MatrixRXd &m1, MatrixRXd &m2) {
#pragma omp parallel for num_threads(1)
    for (int t = 0; t < n; t++) {
        c.row(t) = m1.row(t) * m2.row(t);
    }
}

static void BM_cwise_product(int n, MatrixRXd &m1, MatrixRXd &m2, benchmark_timer::pre_run_info run_info) {
    MatrixRXd c(n, n);
    benchmark_json::benchmarkinfo info = benchmark_timer::benchmark(run_info, cwise_product, c, n, m1, m2);

    benchmark_timer::print_benchmark("Cwise product", n, 0, 0, run_info, info);
}

int main() {
    std::cout << "----- Settings -----" << std::endl;
#ifdef USE_MKL
    std::cout << "Backend = MKL" << std::endl;
#elif USE_ACCELERATE
    std::cout << "Backend = ACCELERATE" << std::endl;
#elif USE_AOCL
    std::cout << "Backend = AOCL" << std::endl;
#else
    std::cout << "Backend = FFTW" << std::endl;
#endif
    std::cout << "----- Settings -----" << std::endl
              << std::endl;

    benchmark_timer::print_header();

    rapidcsv::Document doc("input.csv");

    std::vector<int> runs = doc.GetColumn<int>("run");
    std::vector<std::string> hashes = doc.GetColumn<std::string>("hash");
    std::vector<std::string> functions = doc.GetColumn<std::string>("function");
    std::vector<int> ns = doc.GetColumn<int>("n");
    std::vector<int> bs = doc.GetColumn<int>("b");
    std::vector<int> ds = doc.GetColumn<int>("d");
    std::vector<double> densities = doc.GetColumn<double>("density");
    std::vector<int> matrix_ids = doc.GetColumn<int>("matrix_id");
    std::vector<unsigned int> matrix_seeds = doc.GetColumn<unsigned int>("matrix_seed");
    std::vector<unsigned int> hash_seeds = doc.GetColumn<unsigned int>("hash_seed");
    std::vector<int> sampless = doc.GetColumn<int>("samples");
    std::vector<int> warmup_iterationss = doc.GetColumn<int>("warmup_iterations");
    // std::vector<int> warmup_times = doc.GetColumn<int>("warmup_time");

    int number_of_lines = bs.size();

    std::vector<int> ids;
    std::vector<MatrixRXd> m1s;
    std::vector<MatrixRXd> m2s;
    for (int index = 0; index < number_of_lines; index++) {
        int run = runs[index];
        std::string s_hash = hashes[index];
        std::string s_function = functions[index];
        int n = ns[index];
        int b = bs[index];
        int d = ds[index];
        double density = densities[index];
        int matrix_id = matrix_ids[index];
        unsigned int hash_seed = hash_seeds[index];
        int samples = sampless[index];
        int warmup_iterations = warmup_iterationss[index];
        // int warmup_time = warmup_times[index];

        benchmark_timer::pre_run_info run_info = {
            samples,
            warmup_iterations,
            0,
        };

        if (hash_seed == 0) hash_seed = std::random_device{}();

        if (run == 0) continue;

        if (std::find(ids.begin(), ids.end(), matrix_id) == ids.end()) {
            ids.push_back(matrix_id);

            MatrixRXd temp1(n, n), temp2(n, n);
            std::cin.read(reinterpret_cast<char *>(temp1.data()), n * n * sizeof(double));
            std::cin.read(reinterpret_cast<char *>(temp2.data()), n * n * sizeof(double));

            m1s.push_back(temp1);
            m2s.push_back(temp2);
        }

        int current_matrix_id = std::find(ids.begin(), ids.end(), matrix_id) - ids.begin();

        MatrixRXd m1 = m1s[current_matrix_id];
        MatrixRXd m2 = m2s[current_matrix_id];

        if (s_function == "eigen")
            eigen(m1, m2, run_info);

        if (s_hash == "ful" || s_hash == "full" || s_hash == "random" || s_hash == "rng") {
            FullyRandomHash<int> hash(n, b, d, hash_seed);
            if (s_function == "compress")
                compress<FullyRandomHash<int>>(m1, m2, n, b, d, hash, run_info);
            if (s_function == "compress_th" || s_function == "compress_threaded")
                compress_threaded<FullyRandomHash<int>>(m1, m2, n, b, d, hash, run_info);
            if (s_function == "compress_deluxe")
                compress_deluxe<FullyRandomHash<int>>(m1, m2, n, b, d, hash, run_info);
            if (s_function == "compress_deluxe_threaded" || s_function == "compress_deluxe_th")
                compress_deluxe_threaded<FullyRandomHash<int>>(m1, m2, n, b, d, hash, run_info);
            if (s_function == "decompress")
                decompress<FullyRandomHash<int>>(m1, m2, n, b, d, hash, run_info);
            if (s_function == "both")
                both<FullyRandomHash<int>>(m1, m2, n, b, d, hash, run_info);
        }

        if (s_hash == "mul" || s_hash == "mult" || s_hash == "multiply" || s_hash == "shift") {
            MultiplyShiftHash<uint32_t, uint16_t> hash(d, hash_seed);
            if (s_function == "compress")
                compress<MultiplyShiftHash<uint32_t, uint16_t>>(m1, m2, n, b, d, hash, run_info);
            if (s_function == "compress_th" || s_function == "compress_threaded")
                compress_threaded<MultiplyShiftHash<uint32_t, uint16_t>>(m1, m2, n, b, d, hash, run_info);
            if (s_function == "compress_deluxe")
                compress_deluxe<MultiplyShiftHash<uint32_t, uint16_t>>(m1, m2, n, b, d, hash, run_info);
            if (s_function == "compress_deluxe_threaded" || s_function == "compress_deluxe_th")
                compress_deluxe_threaded<MultiplyShiftHash<uint32_t, uint16_t>>(m1, m2, n, b, d, hash, run_info);
            if (s_function == "decompress")
                decompress<MultiplyShiftHash<uint32_t, uint16_t>>(m1, m2, n, b, d, hash, run_info);
            if (s_function == "both")
                both<MultiplyShiftHash<uint32_t, uint16_t>>(m1, m2, n, b, d, hash, run_info);
        }

        if (s_hash == "tab" || s_hash == "tabulation") {
            TabulationHash<uint32_t, uint32_t, 8> hash(d, hash_seed);
            if (s_function == "compress")
                compress<TabulationHash<uint32_t, uint32_t, 8>>(m1, m2, n, b, d, hash, run_info);
            if (s_function == "compress_th" || s_function == "compress_threaded")
                compress_threaded<TabulationHash<uint32_t, uint32_t, 8>>(m1, m2, n, b, d, hash, run_info);
            if (s_function == "compress_deluxe")
                compress_deluxe<TabulationHash<uint32_t, uint32_t, 8>>(m1, m2, n, b, d, hash, run_info);
            if (s_function == "compress_deluxe_threaded" || s_function == "compress_deluxe_th")
                compress_deluxe_threaded<TabulationHash<uint32_t, uint32_t, 8>>(m1, m2, n, b, d, hash, run_info);
            if (s_function == "decompress")
                decompress<TabulationHash<uint32_t, uint32_t, 8>>(m1, m2, n, b, d, hash, run_info);
            if (s_function == "both")
                both<TabulationHash<uint32_t, uint32_t, 8>>(m1, m2, n, b, d, hash, run_info);
        }

        if (s_function == "eigen_matmul")
            BM_eigen_matmul(n, b, d, m1, m2, run_info);
        if (s_function == "eigen_matmul_par")
            BM_eigen_matmul_par(n, b, d, m1, m2, run_info);
        if (s_function == "matmul")
            BM_matmul(n, b, d, m1, m2, run_info);
        if (s_function == "matmul_par")
            BM_matmul_par(n, b, d, m1, m2, run_info);

        if (s_function == "eigen_array_product")
            BM_eigen_array_product(n, m1, m2, run_info);
        if (s_function == "eigen_cwise_product")
            BM_eigen_cwise_product(n, m1, m2, run_info);
        if (s_function == "cwise_product")
            BM_cwise_product(n, m1, m2, run_info);
    }

    return 0;
}
