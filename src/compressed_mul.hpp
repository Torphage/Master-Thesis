#pragma once
#ifndef COMPRESSED_MUL_HPP
#define COMPRESSED_MUL_HPP

/**
 * @file compressed_mul.hpp
 * @brief Containing functions that compresses and decompresses a matrix multiplication.
 *        Both sequential and parallel versions are available.
 */

#include <fftw3.h>

#include <Eigen/Dense>
#include <complex>
#include <iostream>

#include "fft.hpp"
#include "hashing.hpp"
#include "utils.hpp"

/**
 * @brief Compresses a matrix multiplication according to Pagh's algorithm found in
 *        "Compressed Matrix Multiplication" (2013). This will need to be decompressed in order to
 *        be used. Parameters \p b and \p d change the precision of the final result.
 *
 * @tparam T The type of hash function to use
 * @tparam H The type of data type that the hash function will store its data
 * @tparam Args Let the function take an arbitrary amount of arguments
 * @param m1 is the left matrix
 * @param m2 is the right matrix
 * @param b is at most the number of non-zero elements in the output.
 * @param d is the number of hash functions that will be used
 * @param hash is the type of hash function to use
 * @param hashes is a struct containing each hash function, i.e. h1, h2, s1 and s2.
 * @param args is an arbitrary number of arguments that will be sent to the hash function.
 *             This is used since the hash functions all take in different arguments.
 * @return MatrixRXd Matrix containing the compressed product
 */
template <typename T, typename H, typename... Args>
MatrixRXd compressed_product(const MatrixRXd& m1, const MatrixRXd& m2, int b, int d, T hash, Hashes<H>& hashes, Args... args) {
    int n = m1.rows();

    MatrixRXcd p = MatrixRXcd::Zero(d, b);

    int t, k, i;

    MatrixRXd pa = MatrixRXd::Zero(1, b);
    MatrixRXd pb = MatrixRXd::Zero(1, b);

    Complex* out1 = new Complex[b / 2 + 1];
    Complex* out2 = new Complex[b / 2 + 1];

    fft_struct fft1 = init_fft(b, pa.data(), out1);
    fft_struct fft2 = init_fft(b, pb.data(), out2);

    for (t = 0; t < d; t++) {
        for (k = 0; k < n; k++) {
            pa.setZero();
            pb.setZero();

            for (i = 0; i < n; i++) {
                pa(hash(hashes.h1, t, i, 0, args...)) += hash(hashes.s1, t, i, 1, args...) * m1(i, k);
                pb(hash(hashes.h2, t, i, 0, args...)) += hash(hashes.s2, t, i, 1, args...) * m2(k, i);
            }

            fft(fft1, 0, 0);
            fft(fft2, 0, 0);

            for (i = 0; i < b / 2 + 1; i++) {
                p(t, i) += out1[i] * out2[i];
            }
        }
    }

    clean_fft(fft1);
    clean_fft(fft2);

    MatrixRXd result = MatrixRXd::Zero(d, b);
    ifft_struct info = init_ifft(b, p.data(), result.data());

    for (int t = 0; t < d; t++) {
        ifft(info, t * b, t * b);
    }

    result /= b;
    clean_ifft(info);

    return result;
}

/**
 * @brief Compresses a matrix multiplication in parallel according to Pagh's algorithm found in
 *        "Compressed Matrix Multiplication" (2013). This will need to be decompressed in order to
 *        be used. Parameters \p b and \p d change the precision of the final result.
 *
 * @tparam T The type of hash function to use
 * @tparam H The type of data type that the hash function will store its data
 * @tparam Args Let the function take an arbitrary amount of arguments
 * @param m1 is the left matrix
 * @param m2 is the right matrix
 * @param b is at most the number of non-zero elements in the output.
 * @param d is the number of hash functions that will be used
 * @param hash is the type of hash function to use
 * @param hashes is a struct containing each hash function, i.e. h1, h2, s1 and s2.
 * @param args is an arbitrary number of arguments that will be sent to the hash function.
 *             This is used since the hash functions all take in different arguments.
 * @return MatrixRXd Matrix containing the compressed product
 */
template <typename T, typename H, typename... Args>
MatrixRXd compressed_product_par(const MatrixRXd& m1, const MatrixRXd& m2, int b, int d, T hash, Hashes<H>& hashes, Args... args) {
    int n = m1.rows();

    MatrixRXcd p = MatrixRXcd::Zero(d, b);

    MatrixRXd pas = MatrixRXd::Zero(d, b);
    MatrixRXd pbs = MatrixRXd::Zero(d, b);

    Complex* out1 = new Complex[d * (b / 2 + 1)];
    Complex* out2 = new Complex[d * (b / 2 + 1)];

    fft_struct fft1 = init_fft(b, pas.data(), out1);
    fft_struct fft2 = init_fft(b, pbs.data(), out2);

#pragma omp parallel for
    for (int t = 0; t < d; t++) {
        int in_offset = t * b;
        int out_offset = t * (b / 2 + 1);
        for (int k = 0; k < n; k++) {
            pas.row(t).setZero();
            pbs.row(t).setZero();

            for (int i = 0; i < n; i++) {
                pas(t, hash(hashes.h1, t, i, 0, args...)) += hash(hashes.s1, t, i, 1, args...) * m1(i, k);
                pbs(t, hash(hashes.h2, t, i, 0, args...)) += hash(hashes.s2, t, i, 1, args...) * m2(k, i);
            }

            fft(fft1, in_offset, out_offset);
            fft(fft2, in_offset, out_offset);

            for (int i = 0; i < b / 2 + 1; i++) {
                p(t, i) += out1[i + out_offset] * out2[i + out_offset];
            }
        }
    }

    clean_fft(fft1);
    clean_fft(fft2);

    MatrixRXd result = MatrixRXd::Zero(d, b);
    ifft_struct info = init_ifft(b, p.data(), result.data());

#pragma omp parallel for
    for (int t = 0; t < d; t++) {
        ifft(info, t * b, t * b);
    }

    result /= b;
    clean_ifft(info);

    return result;
}

/**
 * @brief Decompressed a compressed product according to Pagh's algorithm found in
 *        "Compressed Matrix Multiplication" (2013).
 *
 * @tparam T The type of hash function to use
 * @tparam H The type of data type that the hash function will store its data
 * @tparam Args Let the function take an arbitrary amount of arguments
 * @param p is the compressed matrix to be decompressed
 * @param n is the size of the square matrix
 * @param b is at most the number of non-zero elements in the output.
 * @param d is the number of hash functions that will be used
 * @param hash is the type of hash function to use
 * @param hashes is a struct containing each hash function, i.e. h1, h2, s1 and s2.
 * @param args is an arbitrary number of arguments that will be sent to the hash function.
 *             This is used since the hash functions all take in different arguments.
 * @return MatrixRXd Matrix containing the compressed product
 */
template <typename T, typename H, typename... Args>
MatrixRXd decompress_matrix(const MatrixRXd& p, int n, int b, int d, T hash, Hashes<H>& hashes, Args... args) {
    MatrixRXd c = MatrixRXd::Zero(n, n);
    std::vector<double> xt(d);
    double median1;
    double median2;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int t = 0; t < d; t++) {
                xt[t] = hash(hashes.s1, t, i, 1, args...) * hash(hashes.s2, t, j, 1, args...) *
                        p(t, (hash(hashes.h1, t, i, 0, args...) + hash(hashes.h2, t, j, 0, args...)) % b);
            }

            // Median calculations
            std::nth_element(xt.begin(), xt.begin() + d / 2, xt.end());
            median1 = xt[d / 2];

            if (d % 2 != 0) {
                c(i, j) = median1;
            } else {
                std::nth_element(xt.begin(), xt.begin() + (d - 1) / 2, xt.end());
                median2 = xt[d / 2 - 1];
                c(i, j) = (median1 + median2) / 2.0;
            }
        }
    }
    return c;
}

/**
 * @brief Decompressed a compressed product in parallel according to Pagh's algorithm found in
 *        "Compressed Matrix Multiplication" (2013).
 *
 * @tparam T The type of hash function to use
 * @tparam H The type of data type that the hash function will store its data
 * @tparam Args Let the function take an arbitrary amount of arguments
 * @param p is the compressed matrix to be decompressed
 * @param n is the size of the square matrix
 * @param b is at most the number of non-zero elements in the output.
 * @param d is the number of hash functions that will be used
 * @param hash is the type of hash function to use
 * @param hashes is a struct containing each hash function, i.e. h1, h2, s1 and s2.
 * @param args is an arbitrary number of arguments that will be sent to the hash function.
 *             This is used since the hash functions all take in different arguments.
 * @return MatrixRXd Matrix containing the compressed product
 */
template <typename T, typename H, typename... Args>
MatrixRXd decompress_matrix_par(const MatrixRXd& p, int n, int b, int d, T hash, Hashes<H>& hashes, Args... args) {
    MatrixRXd c = MatrixRXd::Zero(n, n);

#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        std::vector<double> xt(d);
        double median1;
        double median2;

        for (int j = 0; j < n; j++) {
            for (int t = 0; t < d; t++) {
                xt[t] = hash(hashes.s1, t, i, 1, args...) * hash(hashes.s2, t, j, 1, args...) *
                        p(t, (hash(hashes.h1, t, i, 0, args...) + hash(hashes.h2, t, j, 0, args...)) % b);
            }

            // Median calculations
            std::nth_element(xt.begin(), xt.begin() + d / 2, xt.end());
            median1 = xt[d / 2];

            if (d % 2 != 0) {
                c(i, j) = median1;
            } else {
                std::nth_element(xt.begin(), xt.begin() + (d - 1) / 2, xt.end());
                median2 = xt[d / 2 - 1];
                c(i, j) = (median1 + median2) / 2.0;
            }
        }
    }
    return c;
}

#endif