#pragma once
#ifndef COMPRESSED_MUL_HPP
#define COMPRESSED_MUL_HPP

#include <complex>
#include <fftw3.h>
#include <Eigen/Dense>
#include <iostream>

#include "hashing.hpp"
#include "utils.hpp"
#include "fft.hpp"

MatrixRXd compressed_ifft(MatrixRXcd& p, int b, int d);

MatrixRXd compressed_ifft_par(MatrixRXcd& p, int b, int d);

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

    MatrixRXd p_real = compressed_ifft(p, b, d);

    return p_real;
}

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

    MatrixRXd p_real = compressed_ifft_par(p, b, d);

    return p_real;
}


/**
 * @brief 
 * 
 * @tparam T 
 * @tparam H 
 * @tparam Args 
 * @param p 
 * @param n 
 * @param b 
 * @param d 
 * @param hash 
 * @param hashes 
 * @param args 
 * @return MatrixRXd 
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