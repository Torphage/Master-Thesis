#include "benchmark_compressed_mul.hpp"
#include "benchmark_json.hpp"

#include "../include/rapidcsv.h"
#include "../src/compressed_mul.hpp"
#include "../src/hashing.hpp"

#include <chrono>
#include <iostream>
#include <random>
#include <unistd.h>

static void eigen(MatrixRXd &m1, MatrixRXd &m2, benchmark_json::config_information &config_info) {
    config_info.name = "Eigen";
    benchmark_timer::benchmark(config_info, [=]() { return (m1.matrix() * m2.matrix()).eval(); });
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

static void BM_eigen_matmul(int n, MatrixRXd &m1, MatrixRXd &m2, benchmark_json::config_information &config_info) {
    MatrixRXd c = MatrixRXd::Zero(n, n);

    config_info.name = "Matmul (Eigen)";
    benchmark_timer::benchmark(config_info, eigen_matmul, c, m1, m2, n);
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

static void BM_eigen_matmul_par(int n, MatrixRXd &m1, MatrixRXd &m2, benchmark_json::config_information &config_info) {
    MatrixRXd c = MatrixRXd::Zero(n, n);

    config_info.name = "Matmul par (Eigen)";
    benchmark_timer::benchmark(config_info, eigen_matmul_par, c, m1, m2, n);
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

static void BM_matmul(int n, MatrixRXd &m1, MatrixRXd &m2, benchmark_json::config_information &config_info) {
    std::vector<std::vector<double>> m1v(n, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> m2v(n, std::vector<double>(n, 0.0));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            m1v[i][j] = m1(i, j);
            m2v[i][j] = m2(i, j);
        }
    }
    std::vector<std::vector<double>> c(n, std::vector<double>(n, 0.0));

    config_info.name = "Matmul";
    benchmark_timer::benchmark(config_info, matmul, c, m1v, m2v, n);
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

static void BM_matmul_par(int n, MatrixRXd &m1, MatrixRXd &m2, benchmark_json::config_information &config_info) {
    std::vector<std::vector<double>> m1v(n, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> m2v(n, std::vector<double>(n, 0.0));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            m1v[i][j] = m1(i, j);
            m2v[i][j] = m2(i, j);
        }
    }
    std::vector<std::vector<double>> c(n, std::vector<double>(n, 0.0));

    config_info.name = "Matmul par";
    benchmark_timer::benchmark(config_info, matmul_par, c, m1v, m2v, n);
}

static void eigen_array_product(MatrixRXd &c, MatrixRXd &m1, MatrixRXd &m2) {
    c = m1 * m2;
}

static void BM_eigen_array_product(int n, MatrixRXd &m1, MatrixRXd &m2, benchmark_json::config_information &config_info) {
    MatrixRXd c(n, n);

    config_info.name = "Eigen Array Coeffwise product";
    benchmark_timer::benchmark(config_info, eigen_array_product, c, m1, m2);
}

static void eigen_cwise_product(MatrixRXd &c, MatrixRXd &m1, MatrixRXd &m2) {
    c = m1.cwiseProduct(m2);
}

static void BM_eigen_cwise_product(int n, MatrixRXd &m1, MatrixRXd &m2, benchmark_json::config_information &config_info) {
    MatrixRXd c(n, n);

    config_info.name = "Eigen Cwise product";
    benchmark_timer::benchmark(config_info, eigen_cwise_product, c, m1, m2);
}

static void cwise_product(MatrixRXd &c, int n, MatrixRXd &m1, MatrixRXd &m2) {
#pragma omp parallel for num_threads(1)
    for (int t = 0; t < n; t++) {
        c.row(t) = m1.row(t) * m2.row(t);
    }
}

static void BM_cwise_product(int n, MatrixRXd &m1, MatrixRXd &m2, benchmark_json::config_information &config_info) {
    MatrixRXd c(n, n);

    config_info.name = "Cwise product";
    benchmark_timer::benchmark(config_info, cwise_product, c, n, m1, m2);
}

int main(int argc, char *argv[]) {
    auto start = std::chrono::steady_clock::now();

    std::cout << "----- Settings -----" << std::endl;
#ifdef USE_MKL
    std::cout << "Backend = MKL" << std::endl;
#elif USE_ACCELERATE
    std::cout << "Backend = ACCELERATE" << std::endl;
#elif USE_AOCL
    std::cout << "Backend = AOCL" << std::endl;
#elif USE_OPENBLAS
    std::cout << "Backend = OPENBLAS" << std::endl;
#else
    std::cout << "Backend = FFTW" << std::endl;
#endif
    std::cout << "----- Settings -----" << std::endl
              << std::endl;

    benchmark_timer::print_header();

    std::string input_file;
    if (argc >= 2) {
        input_file = argv[1];
    } else {
        input_file = "input.csv";
    }
    rapidcsv::Document doc(input_file,
                        rapidcsv::LabelParams(),
                        rapidcsv::SeparatorParams(),
                        rapidcsv::ConverterParams(),
                        rapidcsv::LineReaderParams(true /* pSkipCommentLines */,
                                                    '#' /* pCommentPrefix */,
                                                    true /* pSkipEmptyLines */));


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

    benchmark_json::information info = benchmark_json::information();

    std::vector<int> ids;
    std::vector<MatrixRXd> m1s;
    std::vector<MatrixRXd> m2s;

    for (int index = 0; index < number_of_lines; index++) {
        int run = runs[index];
        int n = ns[index];
        double density = densities[index];
        int matrix_id = matrix_ids[index];
        unsigned int matrix_seed = matrix_seeds[index];

        if (matrix_seed == 0) matrix_seed = std::random_device{}();

        if (run == 0) continue;

        if (std::find(ids.begin(), ids.end(), matrix_id) != ids.end()) {
            continue;
        }

        ids.push_back(matrix_id);
        std::mt19937_64 rng(matrix_seed);

        m1s.push_back(sparse_matrix_generator(n, density, rng));
        m2s.push_back(sparse_matrix_generator(n, density, rng));
    }

    for (int index = 0; index < number_of_lines; index++) {
        int run = runs[index];
        const int n = ns[index];
        const int b = bs[index];
        const int d = ds[index];
        double density = densities[index];
        int matrix_id = matrix_ids[index];
        std::string s_hash = hashes[index];
        std::string s_function = functions[index];
        int matrix_seed = matrix_seeds[index];
        unsigned int hash_seed = hash_seeds[index];
        int samples = sampless[index];
        int warmup_iterations = warmup_iterationss[index];

        if (hash_seed == 0) hash_seed = std::random_device{}();

        if (run == 0) continue;

        benchmark_json::config_information config_info = benchmark_json::config_information();
        config_info.n = n;
        config_info.b = b;
        config_info.d = d;
        config_info.density = density;
        config_info.hash = s_hash;
        config_info.function = s_function;
        config_info.matrix_id = matrix_id;
        config_info.matrix_seed = matrix_seed;
        config_info.hash_seed = hash_seed;
        config_info.samples = samples;
        config_info.warmup_iterations = warmup_iterations;

        auto iter = std::find(matrix_ids.begin(), matrix_ids.end(), matrix_id);
        int new_index = std::distance(matrix_ids.begin(), iter);

        config_info.n = ns[new_index];
        config_info.density = densities[new_index];
        config_info.matrix_id = matrix_ids[new_index];
        config_info.matrix_seed = matrix_seeds[new_index];

        int current_matrix_id = std::find(ids.begin(), ids.end(), matrix_id) - ids.begin();

        MatrixRXd &m1 = m1s[current_matrix_id];
        MatrixRXd &m2 = m2s[current_matrix_id];

        if (s_function == "eigen")
            eigen(m1, m2, config_info);

        if (s_hash == "ful" || s_hash == "full" || s_hash == "random" || s_hash == "rng") {
            FullyRandomHash<int> hash(n, b, d, hash_seed);
            if (s_function == "compress_seq")
                compress_seq<FullyRandomHash<int>>(m1, m2, n, b, d, hash, config_info);
            if (s_function == "compress_par")
                compress_par<FullyRandomHash<int>>(m1, m2, n, b, d, hash, config_info);
            if (s_function == "compress_th" || s_function == "compress_threaded")
                compress_threaded<FullyRandomHash<int>>(m1, m2, n, b, d, hash, config_info);
            if (s_function == "compress_deluxe")
                compress_deluxe<FullyRandomHash<int>>(m1, m2, n, b, d, hash, config_info);
            if (s_function == "compress_secret")
                compress_secret<FullyRandomHash<int>>(m1, m2, n, b, d, hash, config_info);
            if (s_function == "compress_secret2")
                compress_secret2<FullyRandomHash<int>>(m1, m2, n, b, d, hash, config_info);
            if (s_function == "compress_dark")
                compress_dark<FullyRandomHash<int>>(m1, m2, n, b, d, hash, config_info);
            if (s_function == "decompress_seq")
                decompress_seq<FullyRandomHash<int>>(m1, m2, n, b, d, hash, config_info);
            if (s_function == "decompress_par")
                decompress_par<FullyRandomHash<int>>(m1, m2, n, b, d, hash, config_info);
            if (s_function == "decompress_th" || s_function == "decompress_threaded")
                decompress_threaded<FullyRandomHash<int>>(m1, m2, n, b, d, hash, config_info);
            if (s_function == "decompress_dark")
                decompress_dark<FullyRandomHash<int>>(m1, m2, n, b, d, hash, config_info);
            if (s_function == "both")
                both<FullyRandomHash<int>>(m1, m2, n, b, d, hash, config_info);
        }

        if (s_hash == "mul" || s_hash == "mult" || s_hash == "multiply" || s_hash == "shift") {
            MultiplyShiftHash<uint32_t, uint16_t> hash(d, hash_seed);
            if (s_function == "compress_seq")
                compress_seq<MultiplyShiftHash<uint32_t, uint16_t>>(m1, m2, n, b, d, hash, config_info);
            if (s_function == "compress_par")
                compress_par<MultiplyShiftHash<uint32_t, uint16_t>>(m1, m2, n, b, d, hash, config_info);
            if (s_function == "compress_th" || s_function == "compress_threaded")
                compress_threaded<MultiplyShiftHash<uint32_t, uint16_t>>(m1, m2, n, b, d, hash, config_info);
            if (s_function == "compress_deluxe")
                compress_deluxe<MultiplyShiftHash<uint32_t, uint16_t>>(m1, m2, n, b, d, hash, config_info);
            if (s_function == "compress_secret")
                compress_secret<MultiplyShiftHash<uint32_t, uint16_t>>(m1, m2, n, b, d, hash, config_info);
            if (s_function == "compress_secret2")
                compress_secret2<MultiplyShiftHash<uint32_t, uint16_t>>(m1, m2, n, b, d, hash, config_info);
            if (s_function == "compress_dark")
                compress_dark<MultiplyShiftHash<uint32_t, uint16_t>>(m1, m2, n, b, d, hash, config_info);
            if (s_function == "decompress_seq")
                decompress_seq<MultiplyShiftHash<uint32_t, uint16_t>>(m1, m2, n, b, d, hash, config_info);
            if (s_function == "decompress_par")
                decompress_par<MultiplyShiftHash<uint32_t, uint16_t>>(m1, m2, n, b, d, hash, config_info);
            if (s_function == "decompress_th" || s_function == "decompress_threaded")
                decompress_threaded<MultiplyShiftHash<uint32_t, uint16_t>>(m1, m2, n, b, d, hash, config_info);
            if (s_function == "decompress_dark")
                decompress_dark<MultiplyShiftHash<uint32_t, uint16_t>>(m1, m2, n, b, d, hash, config_info);
            if (s_function == "both")
                both<MultiplyShiftHash<uint32_t, uint16_t>>(m1, m2, n, b, d, hash, config_info);
        }

        if (s_hash == "tab" || s_hash == "tabulation") {
            TabulationHash<uint32_t, uint32_t, 8> hash(d, hash_seed);
            if (s_function == "compress_seq")
                compress_seq<TabulationHash<uint32_t, uint32_t, 8>>(m1, m2, n, b, d, hash, config_info);
            if (s_function == "compress_par")
                compress_par<TabulationHash<uint32_t, uint32_t, 8>>(m1, m2, n, b, d, hash, config_info);
            if (s_function == "compress_th" || s_function == "compress_threaded")
                compress_threaded<TabulationHash<uint32_t, uint32_t, 8>>(m1, m2, n, b, d, hash, config_info);
            if (s_function == "compress_deluxe")
                compress_deluxe<TabulationHash<uint32_t, uint32_t, 8>>(m1, m2, n, b, d, hash, config_info);
            if (s_function == "compress_secret")
                compress_secret<TabulationHash<uint32_t, uint32_t, 8>>(m1, m2, n, b, d, hash, config_info);
            if (s_function == "compress_secret2")
                compress_secret2<TabulationHash<uint32_t, uint32_t, 8>>(m1, m2, n, b, d, hash, config_info);
            if (s_function == "compress_dark")
                compress_dark<TabulationHash<uint32_t, uint32_t, 8>>(m1, m2, n, b, d, hash, config_info);
            if (s_function == "decompress_seq")
                decompress_seq<TabulationHash<uint32_t, uint32_t, 8>>(m1, m2, n, b, d, hash, config_info);
            if (s_function == "decompress_par")
                decompress_par<TabulationHash<uint32_t, uint32_t, 8>>(m1, m2, n, b, d, hash, config_info);
            if (s_function == "decompress_th" || s_function == "decompress_threaded")
                decompress_threaded<TabulationHash<uint32_t, uint32_t, 8>>(m1, m2, n, b, d, hash, config_info);
            if (s_function == "decompress_dark")
                decompress_dark<TabulationHash<uint32_t, uint32_t, 8>>(m1, m2, n, b, d, hash, config_info);
            if (s_function == "both")
                both<TabulationHash<uint32_t, uint32_t, 8>>(m1, m2, n, b, d, hash, config_info);
        }

        if (s_function == "eigen_matmul")
            BM_eigen_matmul(n, m1, m2, config_info);
        if (s_function == "eigen_matmul_par")
            BM_eigen_matmul_par(n, m1, m2, config_info);
        if (s_function == "matmul")
            BM_matmul(n, m1, m2, config_info);
        if (s_function == "matmul_par")
            BM_matmul_par(n, m1, m2, config_info);

        if (s_function == "eigen_array_product")
            BM_eigen_array_product(n, m1, m2, config_info);
        if (s_function == "eigen_cwise_product")
            BM_eigen_cwise_product(n, m1, m2, config_info);
        if (s_function == "cwise_product")
            BM_cwise_product(n, m1, m2, config_info);

        info.config.push_back(config_info);
    }

    auto finish = std::chrono::steady_clock::now();

    double total_duration = std::chrono::duration_cast<
                                std::chrono::duration<double>>(finish - start)
                                .count();

    std::cout << "Total benchmarking duration: " << benchmark_timer::suitable_prefix(total_duration).str() << std::endl;

    if (argc >= 3) {
        benchmark_json::write_file(argv[2], info);
    } else {
        benchmark_json::write_file("benchmark.json", info);
    }

    return 0;
}
