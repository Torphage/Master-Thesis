#pragma once
#ifndef VARIANCE_HPP
#define VARIANCE_HPP

#include <vector>
#include <random>
#include <Eigen/Dense>

#include "hashing.hpp"

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

Eigen::MatrixXd variance3d(std::vector<Eigen::MatrixXd> &mat);

bool test_variance(Eigen::MatrixXd m1, Eigen::MatrixXd m2, HashInfo &hashes, int num_samples);

#endif