#ifndef BENCHMARK_COMPRESSED_MUL_HPP_
#define BENCHMARK_COMPRESSED_MUL_HPP_

#include "../include/cxxopts.hpp"
#include "../src/compressed_mul.hpp"
#include "../src/benchmark_timer.hpp"
#include "../src/hashing.hpp"

#include <benchmark/benchmark.h>

#include <iomanip>
#include <iostream>
#include <type_traits>
#include <vector>
#include <string>

template <class T>
static void compress(MatrixRXd& m1, MatrixRXd& m2, int n, int b, int d, T& hash, benchmark_timer::pre_run_info run_info) {
    MatrixRXd compressed = MatrixRXd::Zero(d, b);
    MatrixRXd pas = MatrixRXd::Zero(d, b);
    MatrixRXd pbs = MatrixRXd::Zero(d, b);
    MatrixRXcd p = MatrixRXcd::Zero(d, b);
    MatrixRXcd out1(d, b / 2 + 1);
    MatrixRXcd out2(d, b / 2 + 1);
    fft_struct fft1 = init_fft(b, pas.data(), out1.data());
    fft_struct fft2 = init_fft(b, pbs.data(), out2.data());
    ifft_struct ifft1 = init_ifft(b, p.data(), compressed.data());

    benchmark_timer::benchmarkinfo info = benchmark_timer::benchmark(run_info, bompressed_product_par<T>, m1, m2, b, d, hash, compressed, pas, pbs, p, out1, out2, fft1, fft2, ifft1);

    clean_fft(fft1);
    clean_fft(fft2);
    clean_ifft(ifft1);

    benchmark_timer::print_benchmark("Compress - Original", n, b, d, run_info, info);
}

template <class T>
static void compress_threaded(MatrixRXd& m1, MatrixRXd& m2, int n, int b, int d, T& hash, benchmark_timer::pre_run_info run_info) {
    int num_threads = std::max(d, omp_get_max_threads());
    MatrixRXd compressed = MatrixRXd::Zero(d, b);
    MatrixRXd pas = MatrixRXd::Zero(num_threads, b);
    MatrixRXd pbs = MatrixRXd::Zero(num_threads, b);
    MatrixRXcd p = MatrixRXcd::Zero(d, b);
    MatrixRXcd out1(num_threads, b / 2 + 1);
    MatrixRXcd out2(num_threads, b / 2 + 1);
    fft_struct fft1 = init_fft(b, pas.data(), out1.data());
    fft_struct fft2 = init_fft(b, pbs.data(), out2.data());
    ifft_struct ifft1 = init_ifft(b, p.data(), compressed.data());

    benchmark_timer::benchmarkinfo info = benchmark_timer::benchmark(run_info, bompressed_product_par_threaded<T>, m1, m2, b, d, hash, compressed, pas, pbs, p, out1, out2, fft1, fft2, ifft1);

    clean_fft(fft1);
    clean_fft(fft2);
    clean_ifft(ifft1);

    benchmark_timer::print_benchmark("Compress - Threaded", n, b, d, run_info, info);
}

template <class T>
static void compress_large_threaded(MatrixRXd& m1, MatrixRXd& m2, int n, int b, int d, T& hash, benchmark_timer::pre_run_info run_info) {
    int num_threads = std::max(d, omp_get_max_threads());
    MatrixRXd compressed = MatrixRXd::Zero(d, b);
    MatrixRXd pas = MatrixRXd::Zero(num_threads, b);
    MatrixRXd pbs = MatrixRXd::Zero(num_threads, b);
    MatrixRXcd p = MatrixRXcd::Zero(d, b);
    MatrixRXcd out1(n * d, b / 2 + 1);
    MatrixRXcd out2(n * d, b / 2 + 1);
    fft_struct fft1 = init_fft(b, pas.data(), out1.data());
    fft_struct fft2 = init_fft(b, pbs.data(), out2.data());
    ifft_struct ifft1 = init_ifft(b, p.data(), compressed.data());

    std::cout << "\n\n\nm1: large threaded: \n" << m1 << std::endl;
    std::cout << "m2: large threaded: \n" << m2 << std::endl;

    benchmark_timer::benchmarkinfo info = benchmark_timer::benchmark(run_info, bompressed_product_par_large_threaded<T>, m1, m2, b, d, hash, compressed, pas, pbs, p, out1, out2, fft1, fft2, ifft1);

    clean_fft(fft1);
    clean_fft(fft2);
    clean_ifft(ifft1);

    benchmark_timer::print_benchmark("Compress - Large Threaded", n, b, d, run_info, info);
}

template <class T>
static void compress_large(MatrixRXd& m1, MatrixRXd& m2, int n, int b, int d, T& hash, benchmark_timer::pre_run_info run_info) {
    MatrixRXd compressed = MatrixRXd::Zero(d, b);
    MatrixRXd pas = MatrixRXd::Zero(n * d, b);
    MatrixRXd pbs = MatrixRXd::Zero(n * d, b);
    MatrixRXcd p = MatrixRXcd::Zero(d, b);
    MatrixRXcd out1(n * d, b / 2 + 1);
    MatrixRXcd out2(n * d, b / 2 + 1);
    fft_struct fft1 = init_fft(b, pas.data(), out1.data());
    fft_struct fft2 = init_fft(b, pbs.data(), out2.data());
    ifft_struct ifft1 = init_ifft(b, p.data(), compressed.data());

    benchmark_timer::benchmarkinfo info = benchmark_timer::benchmark(run_info, bompressed_product_par_large<T>, m1, m2, b, d, hash, compressed, pas, pbs, p, out1, out2, fft1, fft2, ifft1);

    clean_fft(fft1);
    clean_fft(fft2);
    clean_ifft(ifft1);

    benchmark_timer::print_benchmark("Compress - Large", n, b, d, run_info, info);
}

template <class T>
static void decompress(MatrixRXd& m1, MatrixRXd& m2, int n, int b, int d, T& hash, benchmark_timer::pre_run_info run_info) {
    MatrixRXd compressed = MatrixRXd::Zero(d, b);
    MatrixRXd pas = MatrixRXd::Zero(d, b);
    MatrixRXd pbs = MatrixRXd::Zero(d, b);
    MatrixRXcd p = MatrixRXcd::Zero(d, b);
    MatrixRXcd out1(d, b / 2 + 1);
    MatrixRXcd out2(d, b / 2 + 1);
    fft_struct fft1 = init_fft(b, pas.data(), out1.data());
    fft_struct fft2 = init_fft(b, pbs.data(), out2.data());
    ifft_struct ifft1 = init_ifft(b, p.data(), compressed.data());

    MatrixRXd result = MatrixRXd::Zero(n, n);
    MatrixRXd xt = MatrixRXd::Zero(n, d);

    bompressed_product_par<T>(m1, m2, b, d, hash, compressed, pas, pbs, p, out1, out2, fft1, fft2, ifft1);
    benchmark_timer::benchmarkinfo info = benchmark_timer::benchmark(run_info, debompress_matrix_par<T>, compressed, n, b, d, hash, result, xt);

    clean_fft(fft1);
    clean_fft(fft2);
    clean_ifft(ifft1);

    benchmark_timer::print_benchmark("Decompress - Original", n, b, d, run_info, info);
}

template <class T>
static void both(MatrixRXd& m1, MatrixRXd& m2, int n, int b, int d, T& hash, benchmark_timer::pre_run_info run_info) {
    MatrixRXd compressed = MatrixRXd::Zero(d, b);
    MatrixRXd pas = MatrixRXd::Zero(d, b);
    MatrixRXd pbs = MatrixRXd::Zero(d, b);
    MatrixRXcd p = MatrixRXcd::Zero(d, b);
    MatrixRXcd out1(d, b / 2 + 1);
    MatrixRXcd out2(d, b / 2 + 1);
    fft_struct fft1 = init_fft(b, pas.data(), out1.data());
    fft_struct fft2 = init_fft(b, pbs.data(), out2.data());
    ifft_struct ifft1 = init_ifft(b, p.data(), compressed.data());

    MatrixRXd result = MatrixRXd::Zero(n, n);
    MatrixRXd xt = MatrixRXd::Zero(n, d);

    benchmark_timer::benchmarkinfo info1 = benchmark_timer::benchmark(run_info, bompressed_product_par<T>, m1, m2, b, d, hash, compressed, pas, pbs, p, out1, out2, fft1, fft2, ifft1);
    benchmark_timer::benchmarkinfo info2 = benchmark_timer::benchmark(run_info, debompress_matrix_par<T>, compressed, n, b, d, hash, result, xt);

    clean_fft(fft1);
    clean_fft(fft2);
    clean_ifft(ifft1);

    benchmark_timer::print_benchmark("Compress - Original", n, b, d, run_info, info1);
    benchmark_timer::print_benchmark("Decompress - Original", n, b, d, run_info, info2);
}

#endif