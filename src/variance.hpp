#pragma once
#ifndef VARIANCE_HPP
#define VARIANCE_HPP

#include <vector>
#include <random>
#include <Eigen/Dense>
#include <iostream>

#include "hashing.hpp"
#include "utils.hpp"

struct HashInfo {
    std::string id;
    std::mt19937_64 rng;
    int b;
    int d;
    int n;
    int p;
    int q;
    int r;
};

double variance(Eigen::VectorXd &vec);

MatrixRXd variance3d(std::vector<MatrixRXd> &mat);

// A template to hold a parameter pack
template < typename... >
struct Typelist {};

template <typename H, typename C, typename T, typename... CArgs, typename... Args>
bool test_variance(MatrixRXd m1, MatrixRXd m2, int num_samples, int b, int d, C constructor, T hash, std::tuple<CArgs...> cargs, Args... args) {    
    int n = m1.rows();
    MatrixRXd compressed;
    MatrixRXd decompressed;

    std::vector<MatrixRXd> vec;
    Hashes<H> hashes;

    for (int i = 0; i < num_samples; i++) {
        hashes = std::apply(constructor, cargs);

        compressed = compressed_product(m1, m2, b, d, hash, hashes, args...);
        decompressed = decompress_matrix(compressed, n, b, d, hash, hashes, args...);

        vec.push_back(decompressed);
        if ((i + 1) % (num_samples / 100) == 0) {
            progress_bar((i + 1.0) / (num_samples));
        }
    }
    MatrixRXd result = variance3d(vec);

    double bound =  (pow((m1*m2).norm(), 2)) / b;

    return (result.array() < bound).all();
}


#endif