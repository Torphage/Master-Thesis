#pragma once
#ifndef COMPRESSED_MUL_HPP
#define COMPRESSED_MUL_HPP

/**
 * @file compressed_mul.hpp
 * @brief Containing functions that compresses and decompresses a matrix multiplication.
 *        Both sequential and parallel versions are available.
 */

#include "fft.hpp"
#include "hashing.hpp"
#include "utils.hpp"

#include <fftw3.h>
#include <Eigen/Dense>
#include <complex>
#include <iostream>
#include <vector>

/*
MatrixRXd& compressed, MatrixRXd& pas, MatrixRXd& pbs, MatrixRXcd& p, MatrixRXcd& out1, MatrixRXcd& out2, fft_struct fft1, fft_struct fft2, ifft_struct ifft
*/

/**
 * @brief Compresses a matrix multiplication according to Pagh's algorithm found in
 *        "Compressed Matrix Multiplication" (2013). This will need to be decompressed in order to
 *        be used. Parameters \p b and \p d change the precision of the final result.
 *
 * @tparam T
 * @param m1 is the left matrix
 * @param m2 is the right matrix
 * @param b is at most the number of non-zero elements in the output.
 * @param d is the number of hash functions that will be used
 * @param hash is
 * @return MatrixRXd Matrix containing the compressed product
 */
template <typename T>
MatrixRXd compressed_product(const MatrixRXd& m1, const MatrixRXd& m2, int b, int d, T hash) {
    int n = m1.rows();

    MatrixRXcd p = MatrixRXcd::Zero(d, b);

    int t, k, i;

    MatrixRXd pa = MatrixRXd::Zero(1, b);
    MatrixRXd pb = MatrixRXd::Zero(1, b);

    std::vector<Complex> out1(b / 2 + 1);
    std::vector<Complex> out2(b / 2 + 1);

    fft_struct fft1 = init_fft(b, pa.data(), out1.data());
    fft_struct fft2 = init_fft(b, pb.data(), out2.data());

    for (t = 0; t < d; t++) {
        for (k = 0; k < n; k++) {
            pa.setZero();
            pb.setZero();

            for (int i = 0; i < n; i++) {
                pa(hash(hash.h1, t, i, b)) += (2 * hash(hash.s1, t, i, 2) - 1) * m1(i, k);
                pb(hash(hash.h2, t, i, b)) += (2 * hash(hash.s2, t, i, 2) - 1) * m2(k, i);
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
 * @tparam T
 * @tparam H The type of data type that the hash function will store its data
 * @tparam Args Allows the function take an arbitrary amount of arguments
 * @param m1 is the left matrix
 * @param m2 is the right matrix
 * @param b is at most the number of non-zero elements in the output.
 * @param d is the number of hash functions that will be used
 * @param hash is
 * @return MatrixRXd Matrix containing the compressed product
 */

template <typename T>
MatrixRXd compressed_product_par(const MatrixRXd& m1, const MatrixRXd& m2, int b, int d, T hash) {
    int n = m1.rows();

    MatrixRXcd p = MatrixRXcd::Zero(d, b);

    MatrixRXd pas = MatrixRXd::Zero(d, b);
    MatrixRXd pbs = MatrixRXd::Zero(d, b);

    std::vector<Complex> out1(d * (b / 2 + 1));
    std::vector<Complex> out2(d * (b / 2 + 1));

    fft_struct fft1 = init_fft(b, pas.data(), out1.data());
    fft_struct fft2 = init_fft(b, pbs.data(), out2.data());

#pragma omp parallel for
    for (int t = 0; t < d; t++) {
        int in_offset = t * b;
        int out_offset = t * (b / 2 + 1);
        for (int k = 0; k < n; k++) {
            pas.row(t).setZero();
            pbs.row(t).setZero();

            for (int i = 0; i < n; i++) {
                pas(t, (hash(hash.h1, t, i, b))) += (2 * hash(hash.s1, t, i, 2) - 1) * m1(i, k);
                pbs(t, (hash(hash.h2, t, i, b))) += (2 * hash(hash.s2, t, i, 2) - 1) * m2(k, i);
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

template <typename T>
void bompressed_product_par(const MatrixRXd& m1, const MatrixRXd& m2, int b, int d, T& hash,
                            MatrixRXd& compressed, MatrixRXd& pas, MatrixRXd& pbs, MatrixRXcd& p,
                            MatrixRXcd& out1, MatrixRXcd& out2, fft_struct fft1, fft_struct fft2, ifft_struct ifft1) {
    int n = m1.rows();

#pragma omp parallel for
    for (int t = 0; t < d; t++) {
        int in_offset = t * b;
        int out_offset = t * (b / 2 + 1);
        for (int k = 0; k < n; k++) {
            pas.row(t).setZero();
            pbs.row(t).setZero();

            for (int i = 0; i < n; i++) {
                // ? Can we get rid of this static_cast? <--- bozo
                pas(t, static_cast<int>(hash(hash.h1, t, i, b))) += (2 * static_cast<int>(hash(hash.s1, t, i, 2)) - 1) * m1(i, k);
                pbs(t, static_cast<int>(hash(hash.h2, t, i, b))) += (2 * static_cast<int>(hash(hash.s2, t, i, 2)) - 1) * m2(k, i);
            }

            fft(fft1, in_offset, out_offset);
            fft(fft2, in_offset, out_offset);

            p.row(t).array() += out1.row(t).array() * out2.row(t).array();
        }
    }

#pragma omp parallel for
    for (int t = 0; t < d; t++) {
        ifft(ifft1, t * (b / 2 + 1), t * b);
    }

    compressed /= b;
}

template <typename T>
void bompressed_product_par_threaded(const MatrixRXd& m1, const MatrixRXd& m2, int b, int d, T& hash,
                                     MatrixRXd& compressed, MatrixRXd& pas, MatrixRXd& pbs, MatrixRXcd& p,
                                     MatrixRXcd& out1, MatrixRXcd& out2, fft_struct fft1, fft_struct fft2, ifft_struct ifft1) {
    int n = m1.rows();

#pragma omp parallel
    {
        int thread_num = omp_get_thread_num();
        int in_offset = thread_num * b;
        int out_offset = thread_num * (b / 2 + 1);
#pragma omp for schedule(dynamic) collapse(2)
        for (int t = 0; t < d; t++) {
            for (int k = 0; k < n; k++) {
                pas.row(thread_num).setZero();
                pbs.row(thread_num).setZero();

                for (int i = 0; i < n; i++) {
                    // ? Can we get rid of this static_cast? <--- bozo
                    pas(thread_num, static_cast<int>(hash(hash.h1, t, i, b))) += (2 * static_cast<int>(hash(hash.s1, t, i, 2)) - 1) * m1(i, k);
                    pbs(thread_num, static_cast<int>(hash(hash.h2, t, i, b))) += (2 * static_cast<int>(hash(hash.s2, t, i, 2)) - 1) * m2(k, i);
                }

                fft(fft1, in_offset, out_offset);
                fft(fft2, in_offset, out_offset);

#pragma omp critical
                p.row(t).array() += out1.row(thread_num).array() * out2.row(thread_num).array();
            }
        }
    }

#pragma omp parallel for
    for (int t = 0; t < d; t++) {
        ifft(ifft1, t * b, t * b);
    }

    compressed /= b;
}

/*
// #pragma omp declare reduction(\
//     add_block : )

// #pragma omp declare reduction(\
//     add_arrays : Eigen::ArrayXd : \
//     omp_out += omp_in \
//   ) initializer(omp_priv = Eigen::ArrayXd::Zero(p_short.row(t).array().size()))
 */

template <typename T>
void bompressed_product_par_threaded2(const MatrixRXd& m1, const MatrixRXd& m2, int b, int d, T& hash,
                                      MatrixRXd& compressed, MatrixRXd& pas, MatrixRXd& pbs, MatrixRXcd& p,
                                      MatrixRXcd& out1, MatrixRXcd& out2, fft_struct fft1, fft_struct fft2, ifft_struct ifft1) {
    int n = m1.rows();

#pragma omp parallel
    {
        int thread_num = omp_get_thread_num();
        int in_offset = thread_num * b;
        int out_offset = thread_num * (b / 2 + 1);
#pragma omp for schedule(dynamic) collapse(2)
        for (int t = 0; t < d; t++) {
            for (int k = 0; k < n; k++) {
                pas.row(thread_num).setZero();
                pbs.row(thread_num).setZero();

                for (int i = 0; i < n; i++) {
                    // ? Can we get rid of this static_cast? <--- bozo
                    pas(thread_num, static_cast<int>(hash(hash.h1, t, i, b))) += (2 * static_cast<int>(hash(hash.s1, t, i, 2)) - 1) * m1(i, k);
                    pbs(thread_num, static_cast<int>(hash(hash.h2, t, i, b))) += (2 * static_cast<int>(hash(hash.s2, t, i, 2)) - 1) * m2(k, i);
                }

                fft(fft1, in_offset, out_offset);
                fft(fft2, in_offset, out_offset);

                // Complex* arr = p_short.row(t).array();
                // #pragma omp critical
                p.row(t).array() += out1.row(thread_num).array() * out2.row(thread_num).array();
            }
        }
    }

#pragma omp parallel for
    for (int t = 0; t < d; t++) {
        ifft(ifft1, t * b, t * b);
    }

    compressed /= b;
}

template <typename T>
void bompressed_product_par_large_threaded(const MatrixRXd& m1, const MatrixRXd& m2, int b, int d, T& hash,
                                           MatrixRXd& compressed, MatrixRXd& pas, MatrixRXd& pbs, MatrixRXcd& p,
                                           MatrixRXcd& out1, MatrixRXcd& out2, fft_struct fft1, fft_struct fft2, ifft_struct ifft1) {
    int n = m1.rows();
    int halfsize = b / 2 + 1;
    int chunk_size = n / omp_get_max_threads();

    for (int t = 0; t < d; t++) {
#pragma omp parallel
        {
            int thread_num = omp_get_thread_num();
            int in_offset = thread_num * b;
#pragma omp for schedule(static, chunk_size)
            for (int k = 0; k < n; k++) {
                pas.row(thread_num).setZero();
                pbs.row(thread_num).setZero();
                int out_offset = (k + t * n) * halfsize;

                for (int i = 0; i < n; i++) {
                    // ? Can we get rid of this static_cast? <--- bozo
                    pas(thread_num, static_cast<int>(hash(hash.h1, t, i, b))) += (2 * static_cast<int>(hash(hash.s1, t, i, 2)) - 1) * m1(i, k);
                    pbs(thread_num, static_cast<int>(hash(hash.h2, t, i, b))) += (2 * static_cast<int>(hash(hash.s2, t, i, 2)) - 1) * m2(k, i);
                }

                fft(fft1, in_offset, out_offset);
                fft(fft2, in_offset, out_offset);
            }
        }
    }

    out1 = out1.cwiseProduct(out2);

#pragma omp parallel for
    for (int t = 0; t < d; t++) {
        p.row(t) = out1.middleRows(t * n, n).colwise().sum().array();
    }

#pragma omp parallel for
    for (int t = 0; t < d; t++) {
        ifft(ifft1, t * halfsize, t * b);
    }

    compressed /= b;
}

template <typename T>
void bompressed_product_par_large_threaded_better(const MatrixRXd& m1, const MatrixRXd& m2, int b, int d, T& hash,
                                                  MatrixRXd& compressed, MatrixRXd& pas, MatrixRXd& pbs, MatrixRXcd& p,
                                                  MatrixRXcd& out1, MatrixRXcd& out2, fft_struct fft1, fft_struct fft2, ifft_struct ifft1) {
    int n = m1.rows();

#pragma omp parallel
    {
        int thread_num = omp_get_thread_num();
        int in_offset = thread_num * b;
#pragma omp for schedule(static) collapse(2)
        for (int t = 0; t < d; t++) {
            for (int k = 0; k < n; k++) {
                pas.row(thread_num).setZero();
                pbs.row(thread_num).setZero();
                int out_offset = (k + t * n) * (b / 2 + 1);

                for (int i = 0; i < n; i++) {
                    // ? Can we get rid of this static_cast? <--- bozo
                    pas(thread_num, static_cast<int>(hash(hash.h1, t, i, b))) += (2 * static_cast<int>(hash(hash.s1, t, i, 2)) - 1) * m1(i, k);
                    pbs(thread_num, static_cast<int>(hash(hash.h2, t, i, b))) += (2 * static_cast<int>(hash(hash.s2, t, i, 2)) - 1) * m2(k, i);
                }

                fft(fft1, in_offset, out_offset);
                fft(fft2, in_offset, out_offset);

//                 int l = std::to_string(out_offset).size();
//                 std::string tmp(8 - l, ' ');
// #pragma omp critical
//                 std::cout << thread_num << ":   " << out_offset << tmp << "t=" << t << " k=" << k << std::endl;
            }
        }
    }

    out1 = out1.cwiseProduct(out2);

#pragma omp parallel for
    for (int t = 0; t < d; t++) {
        p.row(t) = out1.middleRows(t * n, n).colwise().sum().array();
    }

#pragma omp parallel for
    for (int t = 0; t < d; t++) {
        ifft(ifft1, t * (b / 2 + 1), t * b);
    }

    compressed /= b;
}

template <typename T>
void bompressed_product_par_large(const MatrixRXd& m1, const MatrixRXd& m2, int b, int d, T& hash,
                                  MatrixRXd& compressed, MatrixRXd& pas, MatrixRXd& pbs, MatrixRXcd& p,
                                  MatrixRXcd& out1, MatrixRXcd& out2, fft_struct fft1, fft_struct fft2, ifft_struct ifft1) {
    int n = m1.rows();

    // #pragma omp parallel
    //     {
    int index;
    int in_offset;
    int out_offset;
    // #pragma omp for collapse(2)
    for (int t = 0; t < d; t++) {
        for (int k = 0; k < n; k++) {
            index = k + t * n;
            in_offset = index * b;
            out_offset = index * (b / 2 + 1);

            for (int i = 0; i < n; i++) {
                // ? Can we get rid of this static_cast? <--- bozo
                pas(index, static_cast<int>(hash(hash.h1, t, i, b))) += (2 * static_cast<int>(hash(hash.s1, t, i, 2)) - 1) * m1(i, k);
                pbs(index, static_cast<int>(hash(hash.h2, t, i, b))) += (2 * static_cast<int>(hash(hash.s2, t, i, 2)) - 1) * m2(k, i);
            }

            fft(fft1, in_offset, out_offset);
            fft(fft2, in_offset, out_offset);
        }
    }
    // }

    out1 = out1.cwiseProduct(out2);

    for (int t = 0; t < d; t++) {
        p.row(t) = out1.middleRows(t * n, n).colwise().sum().array();
    }

#pragma omp parallel for
    for (int t = 0; t < d; t++) {
        ifft(ifft1, t * b, t * b);
    }

    compressed /= b;
}

/**
 * @brief Decompressed a compressed product according to Pagh's algorithm found in
 *        "Compressed Matrix Multiplication" (2013).
 *
 * @tparam T
 * @param p is the compressed matrix to be decompressed
 * @param n is the size of the square matrix
 * @param b is at most the number of non-zero elements in the output.
 * @param d is the number of hash functions that will be used
 * @param hash is
 * @return MatrixRXd Matrix containing the compressed product
 */
template <typename T>
MatrixRXd decompress_matrix(const MatrixRXd& p, int n, int b, int d, T hash) {
    MatrixRXd c = MatrixRXd::Zero(n, n);
    std::vector<double> xt(d);
    double median1;
    double median2;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int t = 0; t < d; t++) {
                xt[t] = (2 * hash(hash.s1, t, i, 2) - 1) * (2 * hash(hash.s2, t, j, 2) - 1) *
                        p(t, (hash(hash.h1, t, i, b) + hash(hash.h2, t, j, b)) % b);
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
 * @tparam T
 * @param p is the compressed matrix to be decompressed
 * @param n is the size of the square matrix
 * @param b is at most the number of non-zero elements in the output.
 * @param d is the number of hash functions that will be used
 * @param hash is
 * @return MatrixRXd Matrix containing the compressed product
 */
template <typename T>
MatrixRXd decompress_matrix_par(const MatrixRXd& p, int n, int b, int d, T hash) {
    MatrixRXd c = MatrixRXd::Zero(n, n);

#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        std::vector<double> xt(d);
        double median1;
        double median2;

        for (int j = 0; j < n; j++) {
            for (int t = 0; t < d; t++) {
                xt[t] = (2 * hash(hash.s1, t, i, 2) - 1) * (2 * hash(hash.s2, t, j, 2) - 1) *
                        p(t, (hash(hash.h1, t, i, b) + hash(hash.h2, t, j, b)) % b);
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

template <typename T>
void debompress_matrix_par(const MatrixRXd& p, int n, int b, int d, T& hash, MatrixRXd& c, MatrixRXd& xt) {
    double* start = xt.data();

#pragma omp parallel
    {
        double median1;
        double median2;
        double* row;
#pragma omp for schedule(dynamic)
        for (int i = 0; i < n; i++) {
            row = start + (i * d);

            for (int j = 0; j < n; j++) {
                for (int t = 0; t < d; t++) {
                    xt(i, t) = (2 * static_cast<int>(hash(hash.s1, t, i, 2)) - 1) * (2 * static_cast<int>(hash(hash.s2, t, j, 2)) - 1) *
                               p(t, static_cast<int>(hash(hash.h1, t, i, b) + static_cast<int>(hash(hash.h2, t, j, b))) % b);
                }

                // Median calculations
                std::nth_element(row, row + d / 2, row + d);

                median1 = xt(i, d / 2);

                if (d % 2 != 0) {
                    c(i, j) = median1;
                } else {
                    std::nth_element(row, row + (d - 1) / 2, row + d);
                    median2 = xt(i, d / 2 - 1);
                    c(i, j) = (median1 + median2) / 2.0;
                }
            }
        }
    }
}

#endif