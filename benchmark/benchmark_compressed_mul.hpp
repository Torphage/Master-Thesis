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

    MatrixRXd result;
    for (auto _ : state) {
        result = compressed_product_par(m1, m2, b, d, hash);
    };
}

template <typename T>
static void BM_decompress_matrix_par(benchmark::State& state, const MatrixRXd& m1, const MatrixRXd& m2, T hash) {
    int n = m1.rows();
    int b = state.range(0);
    int d = state.range(1);

    MatrixRXd compressed = compressed_product_par(m1, m2, b, d, hash);

    MatrixRXd result;
    for (auto _ : state) {
        result = decompress_matrix_par(compressed, n, b, d, hash);
    };
}

#endif 