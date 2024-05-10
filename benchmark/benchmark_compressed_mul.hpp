#ifndef BENCHMARK_COMPRESSED_MUL_HPP_
#define BENCHMARK_COMPRESSED_MUL_HPP_

#include "../include/cxxopts.hpp"
#include "../src/compressed_mul.hpp"
#include "../src/benchmark_timer.hpp"
#include "../src/hashing.hpp"

#include <iomanip>
#include <iostream>
#include <type_traits>
#include <vector>
#include <string>

template <class T>
static void compress_seq(MatrixRXd& m1, MatrixRXd& m2, int n, int b, int d, T& hash, benchmark_json::config_information& config_info) {
    MatrixRXd compressed = MatrixRXd::Zero(d, b);
    ArrayRXd pa(b);
    MatrixRXcd p = MatrixRXcd::Zero(d, b / 2 + 1);
    Eigen::Array<Complex, 2, Eigen::Dynamic, Eigen::RowMajor> out1(2, b / 2 + 1);
    fft::fft_struct fft1 = fft::init_fft_struct(b, pa.data(), out1.data());
    fft::ifft_struct ifft1 = fft::init_ifft_struct(b, p.data(), compressed.data());

    MatrixRXd m1t = m1.matrix().transpose().array();

    config_info.name = "Compress - Sequential";
    benchmark_timer::benchmark(config_info, bompressed_product_seq<T>, m1t, m2, n, b, d, hash, compressed, pa, p, out1, fft1, ifft1);

    fft::clean_fft(fft1);
    fft::clean_ifft(ifft1);
}

template <class T>
static void compress_par(MatrixRXd& m1, MatrixRXd& m2, int n, int b, int d, T& hash, benchmark_json::config_information& config_info) {
    MatrixRXd compressed = MatrixRXd::Zero(d, b);
    MatrixRXd pas = MatrixRXd::Zero(d, b);
    MatrixRXd pbs = MatrixRXd::Zero(d, b);
    MatrixRXcd p = MatrixRXcd::Zero(d, b / 2 + 1);
    MatrixRXcd out1(d, b / 2 + 1);
    MatrixRXcd out2(d, b / 2 + 1);
    fft::fft_struct fft1 = fft::init_fft_struct(b, pas.data(), out1.data());
    fft::fft_struct fft2 = fft::init_fft_struct(b, pbs.data(), out2.data());
    fft::ifft_struct ifft1 = fft::init_ifft_struct(b, p.data(), compressed.data());

    MatrixRXd m1t = m1.matrix().transpose().array();

    config_info.name = "Compress - Original";
    benchmark_timer::benchmark(config_info, bompressed_product_par<T>, m1t, m2, n, b, d, hash, compressed, pas, pbs, p, out1, out2, fft1, fft2, ifft1);

    fft::clean_fft(fft1);
    fft::clean_fft(fft2);
    fft::clean_ifft(ifft1);
}

template <class T>
static void compress_threaded(MatrixRXd& m1, MatrixRXd& m2, int n, int b, int d, T& hash, benchmark_json::config_information& config_info) {
    int num_threads = omp_get_max_threads();
    MatrixRXd compressed = MatrixRXd::Zero(d, b);
    MatrixRXd pas = MatrixRXd::Zero(num_threads, b);
    MatrixRXd pbs = MatrixRXd::Zero(num_threads, b);
    MatrixRXcd p = MatrixRXcd::Zero(d, b / 2 + 1);
    MatrixRXcd out1(num_threads, b / 2 + 1);
    MatrixRXcd out2(num_threads, b / 2 + 1);
    fft::fft_struct fft1 = fft::init_fft_struct(b, pas.data(), out1.data());
    fft::fft_struct fft2 = fft::init_fft_struct(b, pbs.data(), out2.data());
    fft::ifft_struct ifft1 = fft::init_ifft_struct(b, p.data(), compressed.data());

    MatrixRXd m1t = m1.matrix().transpose().array();

    config_info.name = "Compress - Threaded";
    benchmark_timer::benchmark(config_info, bompressed_product_par_threaded<T>, m1t, m2, n, b, d, hash, compressed, pas, pbs, p, out1, out2, fft1, fft2, ifft1);

    fft::clean_fft(fft1);
    fft::clean_fft(fft2);
    fft::clean_ifft(ifft1);
}

template <class T>
static void compress_deluxe(MatrixRXd& m1, MatrixRXd& m2, int n, int b, int d, T& hash, benchmark_json::config_information& config_info) {
    int num_threads = omp_get_max_threads();
    MatrixRXd compressed = MatrixRXd::Zero(d, b);
    MatrixRXd pas = MatrixRXd::Zero(num_threads, b);
    MatrixRXcd p = MatrixRXcd::Zero(d, b / 2 + 1);
    MatrixRXcd out1(num_threads, b / 2 + 1);
    MatrixRXcd out2(num_threads, b / 2 + 1);
    fft::fft_struct fft1 = fft::init_fft_struct(b, pas.data(), out1.data());
    fft::fft_struct fft2 = fft::init_fft_struct(b, pas.data(), out2.data());
    fft::ifft_struct ifft1 = fft::init_ifft_struct(b, p.data(), compressed.data());

    MatrixRXd m1t = m1.matrix().transpose().array();

    config_info.name = "Compress - Deluxe";
    benchmark_timer::benchmark(config_info, bompressed_product_par_deluxe<T>, m1t, m2, n, b, d, hash, compressed, pas, p, out1, out2, fft1, fft2, ifft1);

    fft::clean_fft(fft1);
    fft::clean_fft(fft2);
    fft::clean_ifft(ifft1);
}

template <class T>
static void compress_secret(MatrixRXd& m1, MatrixRXd& m2, int n, int b, int d, T& hash, benchmark_json::config_information& config_info) {
    int num_threads = omp_get_max_threads();
    int size = std::max(2 * num_threads, d);
    MatrixRXd pas = MatrixRXd::Zero(size, b);
    MatrixRXcd p = MatrixRXcd::Zero(d, b / 2 + 1);
    MatrixRXcd out(4 * num_threads, b / 2 + 1);
    fft::fft_struct fft1 = fft::init_fft_struct(b, pas.data(), out.data());
    fft::ifft_struct ifft1 = fft::init_ifft_struct(b, p.data(), pas.data());

    MatrixRXd m1t = m1.matrix().transpose().array();

    config_info.name = "Compress - Secret Dark Tech";
    benchmark_timer::benchmark(config_info, bompressed_product_par_secret_dark_tech_edition<T>, m1t, m2, n, b, d, hash, pas, p, out, fft1, ifft1);

    fft::clean_fft(fft1);
    fft::clean_ifft(ifft1);
}

template <class T>
static void compress_secret2(MatrixRXd& m1, MatrixRXd& m2, int n, int b, int d, T& hash, benchmark_json::config_information& config_info) {
    int num_threads = omp_get_max_threads();
    int size = std::max(2 * num_threads, d);
    MatrixRXd pas = MatrixRXd::Zero(size, b);
    MatrixRXcd p = MatrixRXcd::Zero(d, b / 2 + 1);
    MatrixRXcd out(4 * num_threads, b / 2 + 1);
    fft::fft_struct fft1 = fft::init_fft_struct(b, pas.data(), out.data());
    fft::ifft_struct ifft1 = fft::init_ifft_struct(b, p.data(), pas.data());

    MatrixRXd m1t = m1.matrix().transpose().array();

    config_info.name = "Compress - Secret Dark Tech 2";
    benchmark_timer::benchmark(config_info, bompressed_product_par_secret_dark_tech_edition2<T>, m1t, m2, n, b, d, hash, pas, p, out, fft1, ifft1);

    fft::clean_fft(fft1);
    fft::clean_ifft(ifft1);
}

template <class T>
static void compress_dark(MatrixRXd& m1, MatrixRXd& m2, int n, int b, int d, T& hash, benchmark_json::config_information& config_info) {
    int b2 = b / 2 + 1;

    MatrixRXd compressed = MatrixRXd::Zero(d, b);
    MatrixRXcd p = MatrixRXcd::Zero(d, b2);
    MatrixRX2i hashes1 = MatrixRX2i::Zero(n, 2);
    MatrixRX2i hashes2 = MatrixRX2i::Zero(n, 2);
    MatrixRXcd out(d, b2);
    fft::fft_plan fft1 = fft::init_fft(b, compressed.data(), out.data());
    fft::fft_plan ifft1 = fft::init_ifft(b, p.data(), compressed.data());

    MatrixRXd m1t = m1.matrix().transpose().array();

    config_info.name = "Compress - Full Darkness";
    benchmark_timer::benchmark(config_info, bompressed_product_par_dark<T>, m1t, m2, n, b, d, hash, compressed, p, hashes1, hashes2, fft1, ifft1);

    fft::clean(fft1);
    fft::clean(ifft1);
}

template <class T>
static void decompress_seq(MatrixRXd& m1, MatrixRXd& m2, int n, int b, int d, T& hash, benchmark_json::config_information& config_info) {
    MatrixRXd compressed = MatrixRXd::Zero(d, b);
    MatrixRXd pas = MatrixRXd::Zero(d, b);
    MatrixRXd pbs = MatrixRXd::Zero(d, b);
    MatrixRXcd p = MatrixRXcd::Zero(d, b / 2 + 1);
    MatrixRXcd out1(d, b / 2 + 1);
    MatrixRXcd out2(d, b / 2 + 1);
    fft::fft_struct fft1 = fft::init_fft_struct(b, pas.data(), out1.data());
    fft::fft_struct fft2 = fft::init_fft_struct(b, pbs.data(), out2.data());
    fft::ifft_struct ifft1 = fft::init_ifft_struct(b, p.data(), compressed.data());

    MatrixRXd m1t = m1.matrix().transpose().array();

    MatrixRXd result = MatrixRXd::Zero(n, n);
    Eigen::ArrayXd xt = Eigen::ArrayXd::Zero(d);

    config_info.name = "Decompress - Sequential";
    bompressed_product_par<T>(m1t, m2, n, b, d, hash, compressed, pas, pbs, p, out1, out2, fft1, fft2, ifft1);
    benchmark_timer::benchmark(config_info, debompress_matrix_seq<T>, compressed, n, b, d, hash, result, xt);

    fft::clean_fft(fft1);
    fft::clean_fft(fft2);
    fft::clean_ifft(ifft1);
}

template <class T>
static void decompress_par(MatrixRXd& m1, MatrixRXd& m2, int n, int b, int d, T& hash, benchmark_json::config_information& config_info) {
    MatrixRXd compressed = MatrixRXd::Zero(d, b);
    MatrixRXd pas = MatrixRXd::Zero(d, b);
    MatrixRXd pbs = MatrixRXd::Zero(d, b);
    MatrixRXcd p = MatrixRXcd::Zero(d, b / 2 + 1);
    MatrixRXcd out1(d, b / 2 + 1);
    MatrixRXcd out2(d, b / 2 + 1);
    fft::fft_struct fft1 = fft::init_fft_struct(b, pas.data(), out1.data());
    fft::fft_struct fft2 = fft::init_fft_struct(b, pbs.data(), out2.data());
    fft::ifft_struct ifft1 = fft::init_ifft_struct(b, p.data(), compressed.data());

    MatrixRXd m1t = m1.matrix().transpose().array();

    MatrixRXd result = MatrixRXd::Zero(n, n);
    MatrixRXd xt = MatrixRXd::Zero(n, d);

    config_info.name = "Decompress - Par";
    bompressed_product_par<T>(m1t, m2, n, b, d, hash, compressed, pas, pbs, p, out1, out2, fft1, fft2, ifft1);
    benchmark_timer::benchmark(config_info, debompress_matrix_par<T>, compressed, n, b, d, hash, result, xt);

    fft::clean_fft(fft1);
    fft::clean_fft(fft2);
    fft::clean_ifft(ifft1);
}

template <class T>
static void decompress_threaded(MatrixRXd& m1, MatrixRXd& m2, int n, int b, int d, T& hash, benchmark_json::config_information& config_info) {
    MatrixRXd compressed = MatrixRXd::Zero(d, b);
    MatrixRXd pas = MatrixRXd::Zero(d, b);
    MatrixRXd pbs = MatrixRXd::Zero(d, b);
    MatrixRXcd p = MatrixRXcd::Zero(d, b / 2 + 1);
    MatrixRXcd out1(d, b / 2 + 1);
    MatrixRXcd out2(d, b / 2 + 1);
    fft::fft_struct fft1 = fft::init_fft_struct(b, pas.data(), out1.data());
    fft::fft_struct fft2 = fft::init_fft_struct(b, pbs.data(), out2.data());
    fft::ifft_struct ifft1 = fft::init_ifft_struct(b, p.data(), compressed.data());

    MatrixRXd m1t = m1.matrix().transpose().array();

    MatrixRXd result = MatrixRXd::Zero(n, n);
    MatrixRXd xt = MatrixRXd::Zero(omp_get_max_threads(), d);

    config_info.name = "Decompress - Threaded";
    bompressed_product_par<T>(m1t, m2, n, b, d, hash, compressed, pas, pbs, p, out1, out2, fft1, fft2, ifft1);
    benchmark_timer::benchmark(config_info, debompress_matrix_par_threaded<T>, compressed, n, b, d, hash, result, xt);

    fft::clean_fft(fft1);
    fft::clean_fft(fft2);
    fft::clean_ifft(ifft1);
}

template <class T>
static void decompress_dark(MatrixRXd& m1, MatrixRXd& m2, int n, int b, int d, T& hash, benchmark_json::config_information& config_info) {
    MatrixRXd compressed = MatrixRXd::Zero(d, b);
    MatrixRXd pas = MatrixRXd::Zero(d, b);
    MatrixRXd pbs = MatrixRXd::Zero(d, b);
    MatrixRXcd p = MatrixRXcd::Zero(d, b / 2 + 1);
    MatrixRXcd out1(d, b / 2 + 1);
    MatrixRXcd out2(d, b / 2 + 1);
    fft::fft_struct fft1 = fft::init_fft_struct(b, pas.data(), out1.data());
    fft::fft_struct fft2 = fft::init_fft_struct(b, pbs.data(), out2.data());
    fft::ifft_struct ifft1 = fft::init_ifft_struct(b, p.data(), compressed.data());

    MatrixRXd m1t = m1.matrix().transpose().array();

    MatrixRXd result(n, n);

    config_info.name = "Decompress - Dark";
    bompressed_product_par<T>(m1t, m2, n, b, d, hash, compressed, pas, pbs, p, out1, out2, fft1, fft2, ifft1);
    benchmark_timer::benchmark(config_info, debompress_matrix_par_dark<T>, compressed, n, b, d, hash, result);

    fft::clean_fft(fft1);
    fft::clean_fft(fft2);
    fft::clean_ifft(ifft1);
}

template <class T>
static void both(MatrixRXd& m1, MatrixRXd& m2, int n, int b, int d, T& hash, benchmark_json::config_information& config_info) {
    MatrixRXd compressed = MatrixRXd::Zero(d, b);
    MatrixRXd pas = MatrixRXd::Zero(d, b);
    MatrixRXd pbs = MatrixRXd::Zero(d, b);
    MatrixRXcd p = MatrixRXcd::Zero(d, b / 2 + 1);
    MatrixRXcd out1(d, b / 2 + 1);
    MatrixRXcd out2(d, b / 2 + 1);
    fft::fft_struct fft1 = fft::init_fft_struct(b, pas.data(), out1.data());
    fft::fft_struct fft2 = fft::init_fft_struct(b, pbs.data(), out2.data());
    fft::ifft_struct ifft1 = fft::init_ifft_struct(b, p.data(), compressed.data());

    MatrixRXd m1t = m1.matrix().transpose().array();

    MatrixRXd result = MatrixRXd::Zero(n, n);
    MatrixRXd xt = MatrixRXd::Zero(n, d);

    // Make a copy of the config_info data type, to use for the other benchmark
    benchmark_json::config_information& config_info2 = config_info;

    config_info.name = "Compress - Original";
    config_info2.name = "Decompress - Original";
    benchmark_timer::benchmark(config_info, bompressed_product_par<T>, m1t, m2, n, b, d, hash, compressed, pas, pbs, p, out1, out2, fft1, fft2, ifft1);
    benchmark_timer::benchmark(config_info2, debompress_matrix_par<T>, compressed, n, b, d, hash, result, xt);

    fft::clean_fft(fft1);
    fft::clean_fft(fft2);
    fft::clean_ifft(ifft1);
}

#endif