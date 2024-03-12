#pragma once
#ifndef VARIANCE_HPP
#define VARIANCE_HPP

#include "hashing.hpp"
#include "utils.hpp"

#include <Eigen/Dense>
#include <iostream>
#include <random>

/**
 * @brief Calculates the variance over a vector
 *
 * @param vec is the vector that the variance will be calculated over
 * @return double The variance
 */
double variance(Eigen::VectorXd &vec);

/**
 * @brief Calculates the variance from a 3d matrix (tensor) projected onto a 2d matrix
 *
 * @param mat The list of matrices to
 * @return MatrixRXd The matrix where each element is the variance of
 *         that position (across the vector)
 */
MatrixRXd variance3d(std::vector<MatrixRXd> &mat);


/**
 * @brief Checks if the variance of a hash function holds below a certain bound. This version will start
 *        checking after a certain number of samples, and continue increasing the number of samples until
 *        the result is valid or until the max number of samples are reached.
 *
 * @tparam T The type of hash function to use
 * @tparam C 
 * @param m1 is the left matrix
 * @param m2 is the right matrix
 * @param num_samples is how many samples start with when calculating the variance
 * @param max_samples is the max number of samples to use when calculating the variance
 * @param b is at most the number of non-zero elements in the output
 * @param d is the number of hash functions
 * @param constructor is the lambda function that constructs a hash of type \p T
 * @return true if the variance holds
 * @return false if the variance does not hold
 */
template <typename T, typename C>
bool test_variance(MatrixRXd m1, MatrixRXd m2, int num_samples, int max_samples, int b, int d, C&& constructor) {
    int n = m1.rows();
    int num_iterations = num_samples;
    int num_samples_used = 0;
    MatrixRXd compressed;
    MatrixRXd decompressed;

    std::vector<MatrixRXd> vec;
    MatrixRXd result = MatrixRXd::Zero(n, n);

    double bound = (pow((m1 * m2).norm(), 2)) / b;

    std::cout << "Iterations running " << num_iterations;
    std::cout.flush();

    while (1) {
        for (int i = 0; i < (max_samples / 10); i++) {
            
            T hash = constructor(std::random_device{}());

            compressed = compressed_product(m1, m2, b, d, hash);
            decompressed = decompress_matrix(compressed, n, b, d, hash);

            vec.push_back(decompressed);
        }
        result = variance3d(vec);

        num_samples_used += (max_samples / 10);
        if ((result.array() < bound).all()) {
            break;
        } else if (num_samples_used >= max_samples) {
            break;
        }

        std::cout << " + " << num_iterations;
        std::cout.flush();
    }

    std::cout << std::endl << "Number of samples used: " << num_samples_used << std::endl;

    return (result.array() < bound).all();
}

#endif