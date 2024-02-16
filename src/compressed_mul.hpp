#include <Eigen/Dense>
#include <complex>
#include <fftw3.h>
#include "hashing.hpp"


typedef std::complex<double> Complex;

/**
 * @brief 
 * 
 * @param m1 
 * @param m2 
 * @param hashes 
 * @return Eigen::MatrixXd 
 */
Eigen::MatrixXd compressed_product(const Eigen::MatrixXd &m1, const Eigen::MatrixXd &m2, BaseHash &hashes);

/**
 * @brief 
 * 
 * @param p 
 * @param n 
 * @param hashes 
 * @return Eigen::MatrixXd 
 */
Eigen::MatrixXd decompress_matrix(Eigen::MatrixXd p, int n, BaseHash &hashes);

/**
 * @brief Calculates the median of a Eigen vector
 * 
 * @param vec A dynamic Eigen vector with doubles 
 * @return The median, with type double
 */
double find_median(Eigen::VectorXd vec);