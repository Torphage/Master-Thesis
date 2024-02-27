#pragma once
#ifndef COMPRESSED_MUL_HPP
#define COMPRESSED_MUL_HPP


#include <Eigen/Dense>
#include <complex>
#include <fftw3.h>

#include "hashing.hpp"
#include "utils.hpp"


/**
 * @brief 
 * 
 * @param m1 
 * @param m2 
 * @param hashes 
 * @return MatrixRXd 
 */
MatrixRXd compressed_product(const MatrixRXd &m1, const MatrixRXd &m2, BaseHash &hashes);


MatrixRXd compressed_product_par(const MatrixRXd &m1, const MatrixRXd &m2, BaseHash &hashes);

/**
 * @brief 
 * 
 * @param p 
 * @param n 
 * @param hashes 
 * @return MatrixRXd 
 */
MatrixRXd decompress_matrix(MatrixRXd p, int n, BaseHash &hashes);

MatrixRXd decompress_matrix_par(MatrixRXd p, int n, BaseHash &hashes);

#endif