#pragma once
#ifndef VARIANCE_HPP
#define VARIANCE_HPP

#include <Eigen/Dense>
#include <iostream>

#include "hashing.hpp"
#include "utils.hpp"

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
 * @brief Checks if the variance of a hash function holds below a certain bound
 *
 * @tparam H The type of data type that the hash function will store its data
 * @tparam C The type of data type that the hash constructor function will store its data
 * @tparam T The type of hash function to use
 * @tparam CArgs A tuple of arbitrary amount of arguments, to be used by the hash constructor
 * @tparam Args Allows the function take an arbitrary amount of arguments
 * @param m1 is the left matrix
 * @param m2 is the right matrix
 * @param num_samples how many samples to calculate the variance over
 * @param b is at most the number of non-zero elements in the output
 * @param d is the number of hash functions
 * @param constructor is the constructor of the hash function to use
 * @param hash is the hash function to use
 * @param cargs is a tuple of arguments sent to the hash constructor
 * @param args are additional optional arguments sent to the hash function
 * @return true if the variance holds
 * @return false if the variance does not hold
 */
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

    double bound = (pow((m1 * m2).norm(), 2)) / b;

    return (result.array() < bound).all();
}

/**
 * @brief Checks if the variance of a hash function holds below a certain bound. This version will start
 *        checking after a certain number of samples, and continue increasing the number of samples until
 *        the result is valid or until the max number of samples are reached.
 *
 * @tparam H The type of data type that the hash function will store its data
 * @tparam C The type of data type that the hash constructor function will store its data
 * @tparam T The type of hash function to use
 * @tparam CArgs A tuple of arbitrary amount of arguments, to be used by the hash constructor
 * @tparam Args Allows the function take an arbitrary amount of arguments
 * @param m1 is the left matrix
 * @param m2 is the right matrix
 * @param num_samples is how many samples start with when calculating the variance
 * @param max_samples is the max number of samples to use when calculating the variance
 * @param b is at most the number of non-zero elements in the output
 * @param d is the number of hash functions
 * @param constructor is the constructor of the hash function to use
 * @param hash is the hash function to use
 * @param cargs is a tuple of arguments sent to the hash constructor
 * @param args are additional optional arguments sent to the hash function
 * @return true if the variance holds
 * @return false if the variance does not hold
 */
template <typename H, typename C, typename T, typename... CArgs, typename... Args>
bool test_variance2(MatrixRXd m1, MatrixRXd m2, int num_samples, int max_samples, int b, int d, C constructor, T hash, std::tuple<CArgs...> cargs, Args... args) {
    int n = m1.rows();
    int num_iterations = num_samples;
    int num_samples_used = 0;
    MatrixRXd compressed;
    MatrixRXd decompressed;

    std::vector<MatrixRXd> vec;
    Hashes<H> hashes;
    MatrixRXd result = MatrixRXd::Zero(n, n);

    double bound = (pow((m1 * m2).norm(), 2)) / b;

    std::cout << "Iterations running " << num_iterations;
    std::cout.flush();

    while (1) {
        num_samples_used += num_iterations;
        for (int i = 0; i < num_iterations; i++) {
            hashes = std::apply(constructor, cargs);

            compressed = compressed_product(m1, m2, b, d, hash, hashes, args...);
            decompressed = decompress_matrix(compressed, n, b, d, hash, hashes, args...);

            vec.push_back(decompressed);
        }
        result = variance3d(vec);
        if ((result.array() < bound).all()) {
            break;
        } else if (num_samples_used >= max_samples) {
            break;
        }

        if (num_iterations >= max_samples / 5) {
            num_iterations += max_samples / 20;
        } else {
            num_iterations += max_samples / 10;
        }

        if (num_samples_used + num_iterations > max_samples)
            num_iterations = max_samples - num_samples_used;

        std::cout << " + " << num_iterations;
        std::cout.flush();
    }

    std::cout << std::endl << "Number of samples used: " << num_samples_used << std::endl;

    return (result.array() < bound).all();
}

#endif