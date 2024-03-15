#ifndef BENCHMARK_COMPRESSED_MUL_HPP_
#define BENCHMARK_COMPRESSED_MUL_HPP_

#include "../include/cxxopts.hpp"
#include "../src/compressed_mul.hpp"
#include "../src/hashing.hpp"

#include <benchmark/benchmark.h>

#include <iostream>
#include <random>

template <typename T>
static void BM_compressed_product_par(benchmark::State& state, const MatrixRXd& m1, const MatrixRXd& m2, T hash) {
    int n = m1.rows();
    int b = state.range(0);
    int d = state.range(1);


    MatrixRXd compressed = MatrixRXd::Zero(d, b);
    MatrixRXd pas = MatrixRXd::Zero(d, b);
    MatrixRXd pbs = MatrixRXd::Zero(d, b);
    MatrixRXcd p = MatrixRXcd::Zero(d, b);
    MatrixRXcd out1(d, b / 2 + 1);
    MatrixRXcd out2(d, b / 2 + 1);
    MatrixRXd result = MatrixRXd::Zero(n, n);
    fft_struct fft1 = init_fft(b, pas.data(), out1.data());
    fft_struct fft2 = init_fft(b, pbs.data(), out2.data());
    ifft_struct ifft = init_ifft(b, p.data(), compressed.data());

    for (auto _ : state) {
        bompressed_product_par(m1, m2, b, d, hash, compressed, pas, pbs, p, out1, out2, fft1, fft2, ifft);
    }

    clean_fft(fft1);
    clean_fft(fft2);
    clean_ifft(ifft);
}

template <typename T>
static void BM_decompress_matrix_par(benchmark::State& state, const MatrixRXd& m1, const MatrixRXd& m2, T hash) {
    int n = m1.rows();
    int b = state.range(0);
    int d = state.range(1);

    MatrixRXd compressed = MatrixRXd::Zero(d, b);
    MatrixRXd pas = MatrixRXd::Zero(d, b);
    MatrixRXd pbs = MatrixRXd::Zero(d, b);
    MatrixRXcd p = MatrixRXcd::Zero(d, b);
    MatrixRXcd out1(d, b / 2 + 1);
    MatrixRXcd out2(d, b / 2 + 1);
    fft_struct fft1 = init_fft(b, pas.data(), out1.data());
    fft_struct fft2 = init_fft(b, pbs.data(), out2.data());
    ifft_struct ifft = init_ifft(b, p.data(), compressed.data());

    bompressed_product_par(m1, m2, b, d, hash, compressed, pas, pbs, p, out1, out2, fft1, fft2, ifft);

    MatrixRXd result = MatrixRXd::Zero(n, n);
    MatrixRXd xt = MatrixRXd::Zero(n, d);

    for (auto _ : state) {
        debompress_matrix_par(compressed, n, b, d, hash, result, xt);
    };

    clean_fft(fft1);
    clean_fft(fft2);
    clean_ifft(ifft);
}

template <typename Word>
static void BM_compressed_product_random_par(benchmark::State& state, const MatrixRXd& m1, const MatrixRXd& m2, int seed) {
    int n = m1.rows();
    int b = state.range(0);
    int d = state.range(1);

    FullyRandomHash<Word> hash(n, b, d, seed);

    MatrixRXd compressed = MatrixRXd::Zero(d, b);
    MatrixRXd pas = MatrixRXd::Zero(d, b);
    MatrixRXd pbs = MatrixRXd::Zero(d, b);
    MatrixRXcd p = MatrixRXcd::Zero(d, b);
    MatrixRXcd out1(d, b / 2 + 1);
    MatrixRXcd out2(d, b / 2 + 1);
    MatrixRXd result = MatrixRXd::Zero(n, n);
    fft_struct fft1 = init_fft(b, pas.data(), out1.data());
    fft_struct fft2 = init_fft(b, pbs.data(), out2.data());
    ifft_struct ifft = init_ifft(b, p.data(), compressed.data());

    for (auto _ : state) {
        bompressed_product_par(m1, m2, b, d, hash, compressed, pas, pbs, p, out1, out2, fft1, fft2, ifft);
    }

    clean_fft(fft1);
    clean_fft(fft2);
    clean_ifft(ifft);
}

template <typename Word>
static void BM_decompress_matrix_random_par(benchmark::State& state, const MatrixRXd& m1, const MatrixRXd& m2, int seed) {
    int n = m1.rows();
    int b = state.range(0);
    int d = state.range(1);

    FullyRandomHash<Word> hash(n, b, d, seed);


    MatrixRXd compressed = MatrixRXd::Zero(d, b);
    MatrixRXd pas = MatrixRXd::Zero(d, b);
    MatrixRXd pbs = MatrixRXd::Zero(d, b);
    MatrixRXcd p = MatrixRXcd::Zero(d, b);
    MatrixRXcd out1(d, b / 2 + 1);
    MatrixRXcd out2(d, b / 2 + 1);
    fft_struct fft1 = init_fft(b, pas.data(), out1.data());
    fft_struct fft2 = init_fft(b, pbs.data(), out2.data());
    ifft_struct ifft = init_ifft(b, p.data(), compressed.data());

    bompressed_product_par(m1, m2, b, d, hash, compressed, pas, pbs, p, out1, out2, fft1, fft2, ifft);

    MatrixRXd result = MatrixRXd::Zero(n, n);
    MatrixRXd xt = MatrixRXd::Zero(n, d);

    for (auto _ : state) {
        debompress_matrix_par(compressed, n, b, d, hash, result, xt);
    };

    clean_fft(fft1);
    clean_fft(fft2);
    clean_ifft(ifft);
}

template <typename Word, typename SmallWord>
static void BM_compressed_product_multiply_shift_par(benchmark::State& state, const MatrixRXd& m1, const MatrixRXd& m2, int seed) {
    int n = m1.rows();
    int b = state.range(0);
    int d = state.range(1);

    MultiplyShiftHash<Word, SmallWord> hash(d, seed);

    MatrixRXd compressed = MatrixRXd::Zero(d, b);
    MatrixRXd pas = MatrixRXd::Zero(d, b);
    MatrixRXd pbs = MatrixRXd::Zero(d, b);
    MatrixRXcd p = MatrixRXcd::Zero(d, b);
    MatrixRXcd out1(d, b / 2 + 1);
    MatrixRXcd out2(d, b / 2 + 1);
    MatrixRXd result = MatrixRXd::Zero(n, n);
    fft_struct fft1 = init_fft(b, pas.data(), out1.data());
    fft_struct fft2 = init_fft(b, pbs.data(), out2.data());
    ifft_struct ifft = init_ifft(b, p.data(), compressed.data());

    for (auto _ : state) {
        bompressed_product_par(m1, m2, b, d, hash, compressed, pas, pbs, p, out1, out2, fft1, fft2, ifft);
    }

    clean_fft(fft1);
    clean_fft(fft2);
    clean_ifft(ifft);
}

template <typename Word, typename SmallWord>
static void BM_decompress_matrix_multiply_shift_par(benchmark::State& state, const MatrixRXd& m1, const MatrixRXd& m2, int seed) {
    int n = m1.rows();
    int b = state.range(0);
    int d = state.range(1);

    MultiplyShiftHash<Word, SmallWord> hash(d, seed);

    MatrixRXd compressed = MatrixRXd::Zero(d, b);
    MatrixRXd pas = MatrixRXd::Zero(d, b);
    MatrixRXd pbs = MatrixRXd::Zero(d, b);
    MatrixRXcd p = MatrixRXcd::Zero(d, b);
    MatrixRXcd out1(d, b / 2 + 1);
    MatrixRXcd out2(d, b / 2 + 1);
    fft_struct fft1 = init_fft(b, pas.data(), out1.data());
    fft_struct fft2 = init_fft(b, pbs.data(), out2.data());
    ifft_struct ifft = init_ifft(b, p.data(), compressed.data());

    bompressed_product_par(m1, m2, b, d, hash, compressed, pas, pbs, p, out1, out2, fft1, fft2, ifft);

    MatrixRXd result = MatrixRXd::Zero(n, n);
    MatrixRXd xt = MatrixRXd::Zero(n, d);

    for (auto _ : state) {
        debompress_matrix_par(compressed, n, b, d, hash, result, xt);
    };

    clean_fft(fft1);
    clean_fft(fft2);
    clean_ifft(ifft);
}

template <typename WordIn, typename WordOut, int r>
static void BM_compressed_product_tabulation_par(benchmark::State& state, const MatrixRXd& m1, const MatrixRXd& m2, int seed) {
    int n = m1.rows();
    int b = state.range(0);
    int d = state.range(1);

    TabulationHash<WordIn, WordOut, r> hash(d, seed);

    MatrixRXd compressed = MatrixRXd::Zero(d, b);
    MatrixRXd pas = MatrixRXd::Zero(d, b);
    MatrixRXd pbs = MatrixRXd::Zero(d, b);
    MatrixRXcd p = MatrixRXcd::Zero(d, b);
    MatrixRXcd out1(d, b / 2 + 1);
    MatrixRXcd out2(d, b / 2 + 1);
    MatrixRXd result = MatrixRXd::Zero(n, n);
    fft_struct fft1 = init_fft(b, pas.data(), out1.data());
    fft_struct fft2 = init_fft(b, pbs.data(), out2.data());
    ifft_struct ifft = init_ifft(b, p.data(), compressed.data());

    for (auto _ : state) {
        bompressed_product_par(m1, m2, b, d, hash, compressed, pas, pbs, p, out1, out2, fft1, fft2, ifft);
    }

    clean_fft(fft1);
    clean_fft(fft2);
    clean_ifft(ifft);
}

template <typename WordIn, typename WordOut, int r>
static void BM_decompress_matrix_tabulation_par(benchmark::State& state, const MatrixRXd& m1, const MatrixRXd& m2, int seed) {
    int n = m1.rows();
    int b = state.range(0);
    int d = state.range(1);

    TabulationHash<WordIn, WordOut, r> hash(d, seed);

    MatrixRXd compressed = MatrixRXd::Zero(d, b);
    MatrixRXd pas = MatrixRXd::Zero(d, b);
    MatrixRXd pbs = MatrixRXd::Zero(d, b);
    MatrixRXcd p = MatrixRXcd::Zero(d, b);
    MatrixRXcd out1(d, b / 2 + 1);
    MatrixRXcd out2(d, b / 2 + 1);
    fft_struct fft1 = init_fft(b, pas.data(), out1.data());
    fft_struct fft2 = init_fft(b, pbs.data(), out2.data());
    ifft_struct ifft = init_ifft(b, p.data(), compressed.data());

    bompressed_product_par(m1, m2, b, d, hash, compressed, pas, pbs, p, out1, out2, fft1, fft2, ifft);

    MatrixRXd result = MatrixRXd::Zero(n, n);
    MatrixRXd xt = MatrixRXd::Zero(n, d);

    for (auto _ : state) {
        debompress_matrix_par(compressed, n, b, d, hash, result, xt);
    };

    clean_fft(fft1);
    clean_fft(fft2);
    clean_ifft(ifft);
}

#endif