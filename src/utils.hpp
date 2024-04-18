#pragma once
#ifndef UTILS_HPP
#define UTILS_HPP

#include <omp.h>
#include <Eigen/Dense>
#include <chrono>
#include <random>

// namespace utils {

/**
 * @brief Alias to \p std::complex<double>
 */
typedef std::complex<double> Complex;

/**
 * @brief A row-major matrix of dynamic size, with type double
 */
typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixRXd;

/**
 * @brief A row-major matrix of dynamic size, with type std::complex<doubles>
 */
typedef Eigen::Array<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixRXcd;

/**
 * @brief A row-major matrix of dynamic size, with type std::complex<uint64_t>
 */
typedef Eigen::Array<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXui;

typedef Eigen::Array<Complex, 1, Eigen::Dynamic, Eigen::RowMajor> ArrayRXcd;

#pragma omp declare reduction(+ : ArrayRXcd : omp_out += omp_in) \
    initializer(omp_priv = ArrayRXcd::Zero(omp_orig.size()))

#pragma omp declare reduction(+ : MatrixRXcd : omp_out += omp_in) \
    initializer(omp_priv = MatrixRXcd::Zero(omp_orig.rows(), omp_orig.cols()))

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
 * @brief A progress bar in the terminal
 *
 * @param percentage is the percentage of completion
 */
void progress_bar(double percentage);

/**
 * @brief This is simply used to roughly measure time
 */
template <class result_t = std::chrono::milliseconds,
          class clock_t = std::chrono::steady_clock,
          class duration_t = std::chrono::milliseconds>
auto since(std::chrono::time_point<clock_t, duration_t> const &start) {
    return std::chrono::duration_cast<result_t>(clock_t::now() - start);
}

// }

#endif