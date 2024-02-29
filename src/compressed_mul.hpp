#pragma once
#ifndef COMPRESSED_MUL_HPP
#define COMPRESSED_MUL_HPP

#include <fftw3.h>

#include <Eigen/Dense>
#include <complex>

#include "hashing.hpp"
#include "utils.hpp"

MatrixRXd compressed_ifft(const MatrixRXcd& p, int b, int d);

MatrixRXd compressed_ifft_par(MatrixRXcd& p, int b, int d);

template <typename T, typename H, typename... Args>
MatrixRXd compressed_product(const MatrixRXd& m1, const MatrixRXd& m2, int b, int d, T hash, Hashes<H>& hashes, Args... args) {
    int n = m1.rows();

    MatrixRXcd p = MatrixRXcd::Zero(d, b);
    MatrixRXd p_real = MatrixRXd::Zero(d, b);

    int t, k, i;
    Complex d1, d2;

    Eigen::VectorXd pa = Eigen::VectorXd::Zero(b);
    Eigen::VectorXd pb = Eigen::VectorXd::Zero(b);

    // 🅱️alloc
    fftw_complex* out1 = fftw_alloc_complex(b / 2 + 1);
    fftw_complex* out2 = fftw_alloc_complex(b / 2 + 1);

    fftw_plan plan1 = fftw_plan_dft_r2c_1d(b, pa.data(), out1, FFTW_ESTIMATE);
    fftw_plan plan2 = fftw_plan_dft_r2c_1d(b, pb.data(), out2, FFTW_ESTIMATE);

    for (t = 0; t < d; t++) {
        for (k = 0; k < n; k++) {
            pa.setZero();
            pb.setZero();

            for (i = 0; i < n; i++) {
                pa(hash(hashes.h1, t, i, 0, args...)) += hash(hashes.s1, t, i, 1, args...) * m1(i, k);
                pb(hash(hashes.h2, t, i, 0, args...)) += hash(hashes.s2, t, i, 1, args...) * m2(k, i);
            }

            fftw_execute(plan1);
            fftw_execute(plan2);

            for (i = 0; i < b / 2 + 1; i++) {
                d1 = Complex(out1[i][0], out1[i][1]);
                d2 = Complex(out2[i][0], out2[i][1]);
                p(t, i) += d1 * d2;
            }
        }
    }

    fftw_free(out1);
    fftw_free(out2);

    fftw_destroy_plan(plan1);
    fftw_destroy_plan(plan2);

    p_real = compressed_ifft(p, b, d);

    return p_real;
}

template <typename T, typename H, typename... Args>
MatrixRXd compressed_product_par(const MatrixRXd& m1, const MatrixRXd& m2, int b, int d, T hash, Hashes<H>& hashes, Args... args) {
    int n = m1.rows();

    MatrixRXcd p = MatrixRXcd::Zero(d, b);

    // 🅱️alloc

    MatrixRXd pas = MatrixRXd::Zero(d, b);
    MatrixRXd pbs = MatrixRXd::Zero(d, b);

    fftw_complex* out1 = fftw_alloc_complex(d * (b / 2 + 1));
    fftw_complex* out2 = fftw_alloc_complex(d * (b / 2 + 1));

    std::vector<fftw_plan> plans1;
    std::vector<fftw_plan> plans2;

    int offset;
    for (int t = 0; t < d; t++) {
        offset = t * (b / 2 + 1);
        plans1.push_back(fftw_plan_dft_r2c_1d(b, pas.row(t).data(), out1 + offset, FFTW_ESTIMATE));
        plans2.push_back(fftw_plan_dft_r2c_1d(b, pbs.row(t).data(), out2 + offset, FFTW_ESTIMATE));
    }

#pragma omp parallel for
    for (int t = 0; t < d; t++) {
        Complex d1;
        Complex d2;
        int offset = t * (b / 2 + 1);
        for (int k = 0; k < n; k++) {
            pas.row(t).setZero();
            pbs.row(t).setZero();

            for (int i = 0; i < n; i++) {
                pas(t, hash(hashes.h1, t, i, 0, args...)) += hash(hashes.s1, t, i, 1, args...) * m1(i, k);
                pbs(t, hash(hashes.h2, t, i, 0, args...)) += hash(hashes.s2, t, i, 1, args...) * m2(k, i);
            }

            fftw_execute(plans1[t]);
            fftw_execute(plans2[t]);

            for (int i = 0; i < b / 2 + 1; i++) {
                d1 = Complex(out1[i + offset][0], out1[i + offset][1]);
                d2 = Complex(out2[i + offset][0], out2[i + offset][1]);
                p(t, i) += d1 * d2;
            }
        }
    }

    fftw_free(out1);
    fftw_free(out2);

    for (int t = 0; t < d; t++) {
        fftw_destroy_plan(plans1[t]);
        fftw_destroy_plan(plans2[t]);
    }

    MatrixRXd p_real = compressed_ifft_par(p, b, d);

    return p_real;
}


// MatrixRXd compressed_product_par(const MatrixRXd& m1, const MatrixRXd& m2, BaseHash& hashes);

// /**
//  * @brief
//  *
//  * @param p
//  * @param n
//  * @param hashes
//  * @return MatrixRXd
//  */
// MatrixRXd decompress_matrix(MatrixRXd p, int n, BaseHash &hashes);

template <typename T, typename H, typename... Args>
MatrixRXd decompress_matrix(const MatrixRXd& p, int n, int b, int d, T hash, Hashes<H>& hashes, Args... args) {
    std::vector<double> xt(d);
    MatrixRXd c = MatrixRXd::Zero(n, n);

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