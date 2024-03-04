#pragma once
#ifndef UTILS_HPP
#define UTILS_HPP

#include <Eigen/Dense>
#include <random>
#include <chrono>
#include <iostream>

typedef std::complex<double> Complex;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixRXd;
typedef Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixRXcd;

/**
 * @brief Generates a random sparse square matrix
 * 
 * @param n is the width and height of the generated matrix
 * @param density is the wanted sparsity, specifically the percentage
 *                of how many non-zeros that should be generated 
 * @param rng is the random number generator
 * @return A random square sparse matrix
 */
MatrixRXd sparse_matrix_generator(int n, float density, std::mt19937_64 &rng);

/**
 * @brief Rounds the values in a matrix to the n-th decimal.
 * More technically it removes all numbers after the n-th decimal 
 * on a value, for values that are smaller than 10^(-n)
 * 
 * @param matrix is the matrix to be rounded
 * @param n is how many decimals that should be kept
 */
void round_matrix(MatrixRXd &matrix, int n);

/**
 * @brief Calculates the total sum of all the values in a matrix
 * 
 * @param matrix is the matrix to get the values from
 * @return The total sum of all the values in the given matrix
 */
double sum_matrix(MatrixRXd &matrix);

/**
 * @brief Calculates the median of a Eigen vector
 * 
 * @param vec A dynamic Eigen vector with doubles 
 * @return The median, with type double
 */
double find_median(Eigen::VectorXd vec);

void progress_bar(double percentage);

template <
    class result_t   = std::chrono::milliseconds,
    class clock_t    = std::chrono::steady_clock,
    class duration_t = std::chrono::milliseconds
>
auto since(std::chrono::time_point<clock_t, duration_t> const& start)
{
    return std::chrono::duration_cast<result_t>(clock_t::now() - start);
}

#endif