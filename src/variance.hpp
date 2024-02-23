#pragma once
#ifndef VARIANCE_HPP
#define VARIANCE_HPP

#include <vector>
#include <random>
#include <Eigen/Dense>

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

bool test_variance(MatrixRXd m1, MatrixRXd m2, HashInfo &hashes, int num_samples);

#endif