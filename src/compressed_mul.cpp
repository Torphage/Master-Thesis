#include "compressed_mul.hpp"

#include <fftw3.h>
#include <omp.h>

#include <cmath>
#include <complex>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

#include "hashing.hpp"
#include "utils.hpp"

MatrixRXd compressed_ifft(const MatrixRXcd& p, int b, int d) {
    fftw_complex* in = fftw_alloc_complex(b);
    double* out = fftw_alloc_real(2 * (b / 2 + 1));

    fftw_plan plan = fftw_plan_dft_c2r_1d(b, in, out, FFTW_ESTIMATE);
    MatrixRXd p_real = MatrixRXd::Zero(d, b);
    for (int t = 0; t < d; t++) {
        // Copy data from p to in, otherwise p would be overwritten later
        for (int i = 0; i < b; i++) {
            in[i][0] = p(t, i).real();
            in[i][1] = p(t, i).imag();
        }

        fftw_execute(plan);

        for (int i = 0; i < b; i++) {
            p_real(t, i) = out[i] / b;
        }
    }
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return p_real;
}

MatrixRXd compressed_ifft_par(MatrixRXcd& p, int b, int d) {
    fftw_complex* in;

    MatrixRXd result = MatrixRXd::Zero(d, b);
    std::vector<fftw_plan> plans;

    for (int t = 0; t < d; t++) {
        in = reinterpret_cast<fftw_complex*>(p.row(t).data());
        plans.push_back(
            fftw_plan_dft_c2r_1d(b, in, result.row(t).data(), FFTW_ESTIMATE | FFTW_PRESERVE_INPUT));
    }

#pragma omp parallel for
    for (int t = 0; t < d; t++) {
        fftw_execute(plans[t]);
    }

    result /= b;

    for (int t = 0; t < d; t++) {
        fftw_destroy_plan(plans[t]);
    }

    return result;
}

double decompress_element(const MatrixRXd& p, int i, int j, int d, int b, BaseHash& hashes) {
    Eigen::VectorXd xt = Eigen::VectorXd::Zero(d);
    for (int t = 0; t < d; t++) {
        xt(t) = hashes.hash(2, t, i) * hashes.hash(3, t, j) *
                p(t, (hashes.hash(0, t, i) + hashes.hash(1, t, j)) % b);
    }

    return find_median(xt);
}

MatrixRXd decompress_matrix(MatrixRXd p, int n, BaseHash& hashes) {
    int b = hashes.b;
    int d = hashes.d;
    MatrixRXd c = MatrixRXd::Zero(n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            c(i, j) = decompress_element(p, i, j, d, b, hashes);
        }
    }
    return c;
}

MatrixRXd decompress_matrix_par(MatrixRXd p, int n, BaseHash& hashes) {
    int b = hashes.b;
    int d = hashes.d;
    MatrixRXd c = MatrixRXd::Zero(n, n);
#pragma omp parallel for
    for (int i = 0; i < n * n; i++) {
        int row = i / n;
        int col = i % n;
        c(row, col) = decompress_element(p, row, col, d, b, hashes);
    }
    return c;
}
