#pragma once
#ifndef UTILS_HPP
#define UTILS_HPP

#include <Eigen/Dense>
#include <random>

/**
 * @brief Generates a random sparse square matrix
 * 
 * @param n is the width and height of the generated matrix
 * @param density is the wanted sparsity, specifically the percentage
 *                of how many non-zeros that should be generated 
 * @param rng is the random number generator
 * @return A random square sparse matrix
 */
Eigen::MatrixXd sparse_matrix_generator(int n, float density, std::mt19937_64 &rng);

/**
 * @brief Rounds the values in a matrix to the n-th decimal.
 * More technically it removes all numbers after the n-th decimal 
 * on a value, for values that are smaller than 10^(-n)
 * 
 * @param matrix is the matrix to be rounded
 * @param n is how many decimals that should be kept
 */
void round_matrix(Eigen::MatrixXd &matrix, int n);

/**
 * @brief Calculates the total sum of all the values in a matrix
 * 
 * @param matrix is the matrix to get the values from
 * @return The total sum of all the values in the given matrix
 */
double sum_matrix(Eigen::MatrixXd &matrix);


#endif