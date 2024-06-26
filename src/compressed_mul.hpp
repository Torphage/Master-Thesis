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

    fft::fft_struct fft1 = fft::init_fft_struct(b, pa.data(), out1.data());
    fft::fft_struct fft2 = fft::init_fft_struct(b, pb.data(), out2.data());

    for (t = 0; t < d; t++) {
        for (k = 0; k < n; k++) {
            pa.setZero();
            pb.setZero();

            for (int i = 0; i < n; i++) {
                pa(hash(hash.h1, t, i, b)) += (2 * hash(hash.s1, t, i, 2) - 1) * m1(k, i);
                pb(hash(hash.h2, t, i, b)) += (2 * hash(hash.s2, t, i, 2) - 1) * m2(k, i);
            }

            fft::fft(fft1, 0, 0);
            fft::fft(fft2, 0, 0);

            for (i = 0; i < b / 2 + 1; i++) {
                p(t, i) += out1[i] * out2[i];
            }
        }
    }

    fft::clean_fft(fft1);
    fft::clean_fft(fft2);

    MatrixRXd result = MatrixRXd::Zero(d, b);
    fft::ifft_struct info = fft::init_ifft_struct(b, p.data(), result.data());

    for (int t = 0; t < d; t++) {
        ifft(info, t * b, t * b);
    }

    result /= b;
    fft::clean_ifft(info);

    return result;
}

template <typename T>
void bompressed_product_seq(const MatrixRXd& m1, const MatrixRXd& m2, int n, int b, int d, T& hash,
                            MatrixRXd& compressed, ArrayRXd& pa, MatrixRXcd& p,
                            Eigen::Array<Complex, 2, Eigen::Dynamic, Eigen::RowMajor>& out, fft::fft_struct fft1, fft::ifft_struct ifft1) {
    int fft2_offset = b / 2 + 1;
    for (int t = 0; t < d; t++) {
        for (int k = 0; k < n; k++) {
            pa.setZero();
            for (int i = 0; i < n; i++) {
                pa(hash(hash.h1, t, i, b)) += (2 * hash(hash.s1, t, i, 2) - 1) * m1(k, i);
            }
            fft::fft(fft1, 0, 0);

            pa.setZero();
            for (int i = 0; i < n; i++) {
                pa(hash(hash.h2, t, i, b)) += (2 * hash(hash.s2, t, i, 2) - 1) * m2(k, i);
            }
            fft::fft(fft1, 0, fft2_offset);

            p.row(t) += out.row(0) * out.row(1);
        }
    }

    for (int t = 0; t < d; t++) {
        fft::ifft(ifft1, t * fft2_offset, t * b);
    }

    compressed /= b;
}

template <typename T>
void bompressed_product_par(const MatrixRXd& m1, const MatrixRXd& m2, int n, int b, int d, T& hash,
                            MatrixRXd& compressed, MatrixRXd& pas, MatrixRXd& pbs, MatrixRXcd& p,
                            MatrixRXcd& out1, MatrixRXcd& out2, fft::fft_struct fft1, fft::fft_struct fft2, fft::ifft_struct ifft1) {
#pragma omp parallel for
    for (int t = 0; t < d; t++) {
        int in_offset = t * b;
        int out_offset = t * (b / 2 + 1);
        for (int k = 0; k < n; k++) {
            pas.row(t).setZero();
            pbs.row(t).setZero();

            for (int i = 0; i < n; i++) {
                pas(t, static_cast<int>(hash(hash.h1, t, i, b))) += (2 * static_cast<int>(hash(hash.s1, t, i, 2)) - 1) * m1(k, i);
                pbs(t, static_cast<int>(hash(hash.h2, t, i, b))) += (2 * static_cast<int>(hash(hash.s2, t, i, 2)) - 1) * m2(k, i);
            }

            fft::fft(fft1, in_offset, out_offset);
            fft::fft(fft2, in_offset, out_offset);

            p.row(t).array() += out1.row(t).array() * out2.row(t).array();
        }
    }

#pragma omp parallel for
    for (int t = 0; t < d; t++) {
        fft::ifft(ifft1, t * (b / 2 + 1), t * b);
    }

    compressed /= b;
}

template <typename T>
void bompressed_product_par_threaded(const MatrixRXd& m1, const MatrixRXd& m2, int n, int b, int d, T& hash,
                                     MatrixRXd& compressed, MatrixRXd& pas, MatrixRXd& pbs, MatrixRXcd& p,
                                     MatrixRXcd& out1, MatrixRXcd& out2, fft::fft_struct fft1, fft::fft_struct fft2, fft::ifft_struct ifft1) {
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
                    pas(thread_num, static_cast<int>(hash(hash.h1, t, i, b))) += (2 * static_cast<int>(hash(hash.s1, t, i, 2)) - 1) * m1(k, i);
                    pbs(thread_num, static_cast<int>(hash(hash.h2, t, i, b))) += (2 * static_cast<int>(hash(hash.s2, t, i, 2)) - 1) * m2(k, i);
                }

                fft::fft(fft1, in_offset, out_offset);
                fft::fft(fft2, in_offset, out_offset);

#pragma omp critical
                p.row(t).array() += out1.row(thread_num).array() * out2.row(thread_num).array();
            }
        }
    }

#pragma omp parallel for
    for (int t = 0; t < d; t++) {
        fft::ifft(ifft1, t * b, t * b);
    }

    compressed /= b;
}

template <typename T>
void bompressed_product_par_deluxe(const MatrixRXd& m1, const MatrixRXd& m2, int n, int b, int d, T& hash,
                                   MatrixRXd& compressed, MatrixRXd& pas, MatrixRXcd& p, MatrixRXcd& out1,
                                   MatrixRXcd& out2, const fft::fft_struct fft1, const fft::fft_struct fft2, const fft::ifft_struct ifft1) {
#pragma omp parallel
    {
        int thread_num = omp_get_thread_num();
        int in_offset = thread_num * b;
        int out_offset = thread_num * (b / 2 + 1);
#pragma omp for schedule(static) collapse(2) reduction(+ : p)
        for (int t = 0; t < d; t++) {
            for (int k = 0; k < n; k++) {
                pas.row(thread_num).setZero();

                for (int i = 0; i < n; i++) {
                    pas(thread_num, static_cast<int>(hash(hash.h1, t, i, b))) += (2 * static_cast<int>(hash(hash.s1, t, i, 2)) - 1) * m1(k, i);
                }

                fft::fft(fft1, in_offset, out_offset);

                pas.row(thread_num).setZero();
                for (int i = 0; i < n; i++) {
                    pas(thread_num, static_cast<int>(hash(hash.h2, t, i, b))) += (2 * static_cast<int>(hash(hash.s2, t, i, 2)) - 1) * m2(k, i);
                }

                fft::fft(fft2, in_offset, out_offset);

                p.row(t) += out1.row(thread_num) * out2.row(thread_num);
            }
        }
#pragma omp for
        for (int t = 0; t < d; t++) {
            fft::ifft(ifft1, t * (b / 2 + 1), t * b);
            compressed.row(t) /= b;
        }
    }
}

template <typename T>
void bompressed_product_par_secret_dark_tech_edition(const MatrixRXd& m1, const MatrixRXd& m2, int n, int b, int d, T& hash,
                                                     MatrixRXd& pas, MatrixRXcd& p,
                                                     MatrixRXcd& out1, const fft::fft_struct fft1, const fft::ifft_struct ifft1) {
#pragma omp parallel
    {
        int thread_num = 2 * omp_get_thread_num();
        int in_offset = thread_num * b;
        int out_offset = thread_num * (b / 2 + 1);
#pragma omp for schedule(auto) collapse(2) reduction(+ : p)
        for (int t = 0; t < d; t++) {
            for (int k = 0; k < n; k++) {
                pas.row(thread_num).setZero();

                for (int i = 0; i < n; i++) {
                    pas(thread_num, static_cast<int>(hash(hash.h1, t, i, b))) += (2 * static_cast<int>(hash(hash.s1, t, i, 2)) - 1) * m1(k, i);
                }

                fft::fft(fft1, in_offset, out_offset);
                pas.row(thread_num).setZero();

                for (int i = 0; i < n; i++) {
                    pas(thread_num, static_cast<int>(hash(hash.h2, t, i, b))) += (2 * static_cast<int>(hash(hash.s2, t, i, 2)) - 1) * m2(k, i);
                }

                fft::fft(fft1, in_offset, out_offset + 2 * omp_get_max_threads() * (b / 2 + 1));

                p.row(t) += out1.row(thread_num) * out1.row(thread_num + 2 * omp_get_max_threads());
            }
        }
#pragma omp for
        for (int t = 0; t < d; t++) {
            fft::ifft(ifft1, t * (b / 2 + 1), t * b);
            pas.row(t) /= b;
        }
    }
}

template <typename T>
void bompressed_product_par_secret_dark_tech_edition2(const MatrixRXd& m1, const MatrixRXd& m2, int n, int b, int d, T& hash,
                                                      MatrixRXd& pas, MatrixRXcd& p, MatrixRXcd& out1,
                                                      fft::fft_struct fft1, fft::ifft_struct ifft1) {
    std::vector<std::tuple<int, double>> aids1(n);
    std::vector<std::tuple<int, double>> aids2(n);
#pragma omp parallel
    {
        int thread_num = 2 * omp_get_thread_num();
        int in_offset = thread_num * b;
        int out_offset = thread_num * (b / 2 + 1);

        for (int t = 0; t < d; t++) {
            for (int i = 0; i < n; i++) {
                aids1[i] = std::make_tuple((hash(hash.h1, t, i, b)), (2 * (hash(hash.s1, t, i, 2)) - 1));
                aids2[i] = std::make_tuple((hash(hash.h2, t, i, b)), (2 * (hash(hash.s2, t, i, 2)) - 1));
            }

#pragma omp for schedule(auto) reduction(+ : p)
            for (int k = 0; k < n; k++) {
                pas.row(thread_num).setZero();

                for (int i = 0; i < n; i++) {
                    pas(thread_num, std::get<0>(aids1[i])) += std::get<1>(aids1[i]) * m1(k, i);
                }

                fft::fft(fft1, in_offset, out_offset);

                pas.row(thread_num).setZero();

                for (int i = 0; i < n; i++) {
                    pas(thread_num, std::get<0>(aids2[i])) += std::get<1>(aids2[i]) * m2(k, i);
                }

                fft::fft(fft1, in_offset, out_offset + 2 * omp_get_max_threads() * (b / 2 + 1));

                p.row(t) += out1.row(thread_num) * out1.row(thread_num + 2 * omp_get_max_threads());
            }
        }
#pragma omp for
        for (int t = 0; t < d; t++) {
            fft::ifft(ifft1, t * (b / 2 + 1), t * b);
            pas.row(t) /= b;
        }
    }
}

template <typename T>
void bompressed_product_par_dark(const MatrixRXd& m1, const MatrixRXd& m2, const int n, const int b, const int d, T& hash,
                                 MatrixRXd& compressed, MatrixRXcd& p,
                                 fft::fft_plan& fft1, fft::fft_plan& ifft1) {
#pragma omp parallel
    {
        Eigen::ArrayXd pa = Eigen::ArrayXd::Zero(b);
        Eigen::ArrayXcd out1 = Eigen::ArrayXcd::Zero(b / 2 + 1);
        Eigen::ArrayXcd out2 = Eigen::ArrayXcd::Zero(b / 2 + 1);

        for (int t = 0; t < d; t++) {
#pragma omp for schedule(auto) reduction(+ : p)
            for (int k = 0; k < n; k++) {
                pa.setZero();
                for (int i = 0; i < n; i++) {
                    pa(hash(hash.h1, t, i, b)) += (2 * hash(hash.s1, t, i, 2) - 1) * m1(k, i);
                }
                fft::execute_fft(fft1, pa.data(), out1.data());

                pa.setZero();
                for (int i = 0; i < n; i++) {
                    pa(hash(hash.h2, t, i, b)) += (2 * hash(hash.s2, t, i, 2) - 1) * m2(k, i);
                }
                fft::execute_fft(fft1, pa.data(), out2.data());

                p.row(t) += out1 * out2;
            }
        }
#pragma omp for
        for (int t = 0; t < d; t++) {
            fft::execute_ifft(ifft1, p.row(t).data(), compressed.row(t).data());
            compressed.row(t) /= b;
        }
    }
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
                xt[t] = (2 * static_cast<int>(hash(hash.s1, t, i, 2)) - 1) * (2 * static_cast<int>(hash(hash.s2, t, j, 2)) - 1) *
                        p(t, (static_cast<int>(hash(hash.h1, t, i, b)) + static_cast<int>(hash(hash.h2, t, j, b))) % b);
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
void debompress_matrix_seq(const MatrixRXd& p, int n, int b, int d, T& hash, MatrixRXd& result, Eigen::ArrayXd& xt) {
    double* row = xt.data();
    double* middle_odd = row + d / 2;
    double* middle_even = row + (d - 1) / 2;
    double* end = row + d;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int t = 0; t < d; t++) {
                xt(t) = (2 * static_cast<int>(hash(hash.s1, t, i, 2)) - 1) * (2 * static_cast<int>(hash(hash.s2, t, j, 2)) - 1) *
                        p(t, (static_cast<int>(hash(hash.h1, t, i, b)) + static_cast<int>(hash(hash.h2, t, j, b))) % b);
            }

            // Median calculations
            std::nth_element(row, middle_odd, end);

            if (d % 2 != 0) {
                result(i, j) = xt(d / 2);
            } else {
                std::nth_element(row, middle_even, middle_odd);
                result(i, j) = (xt(d / 2) + xt(d / 2 - 1)) / 2.0;
            }
        }
    }
}

template <typename T>
void debompress_matrix_par(const MatrixRXd& p, int n, int b, int d, T& hash, MatrixRXd& c, MatrixRXd& xt) {
    double* start = xt.data();

#pragma omp parallel
    {
        double median1;
        double median2;
        double* row;
#pragma omp for
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
                    std::nth_element(row, row + (d - 1) / 2, row + d / 2);
                    median2 = xt(i, d / 2 - 1);
                    c(i, j) = (median1 + median2) / 2.0;
                }
            }
        }
    }
}

template <typename T>
void debompress_matrix_par_threaded(const MatrixRXd& p, int n, int b, int d, T& hash, MatrixRXd& c, MatrixRXd& xt) {
    double* start = xt.data();

#pragma omp parallel
    {
        int thread_num = omp_get_thread_num();
        double median1;
        double median2;
        double* row;
#pragma omp for
        for (int i = 0; i < n; i++) {
            row = start + (thread_num * d);

            for (int j = 0; j < n; j++) {
                for (int t = 0; t < d; t++) {
                    xt(thread_num, t) = (2 * static_cast<int>(hash(hash.s1, t, i, 2)) - 1) * (2 * static_cast<int>(hash(hash.s2, t, j, 2)) - 1) *
                                        p(t, static_cast<int>(hash(hash.h1, t, i, b) + static_cast<int>(hash(hash.h2, t, j, b))) % b);
                }

                // Median calculations
                std::nth_element(row, row + d / 2, row + d);

                median1 = xt(thread_num, d / 2);

                if (d % 2 != 0) {
                    c(i, j) = median1;
                } else {
                    std::nth_element(row, row + (d - 1) / 2, row + d / 2);
                    median2 = xt(thread_num, d / 2 - 1);
                    c(i, j) = (median1 + median2) / 2.0;
                }
            }
        }
    }
}

template <typename T>
void debompress_matrix_par_dark(const MatrixRXd& p, int n, int b, int d, T& hash, MatrixRXd& result) {
#pragma omp parallel
    {
        Eigen::ArrayXd xt(d);
        double* row = xt.data();
        double* start = row + d / 2;
        double* start2 = row + (d - 1) / 2;
        double* end = row + d;
#pragma omp for
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int t = 0; t < d; t++) {
                    xt(t) = (2 * static_cast<int>(hash(hash.s1, t, i, 2)) - 1) * (2 * static_cast<int>(hash(hash.s2, t, j, 2)) - 1) *
                            p(t, (static_cast<int>(hash(hash.h1, t, i, b)) + static_cast<int>(hash(hash.h2, t, j, b))) % b);
                }

                // Median calculations
                std::nth_element(row, start, end);
                double median1 = xt(d / 2);

                if (d % 2 != 0) {
                    result(i, j) = median1;
                } else {
                    std::nth_element(row, start2, start);
                    double median2 = xt(d / 2 - 1);
                    result(i, j) = (median1 + median2) / 2.0;
                }
            }
        }
    }
}

#endif