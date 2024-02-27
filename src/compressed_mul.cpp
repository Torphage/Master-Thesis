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

MatrixRXd compressed_product(const MatrixRXd& m1, const MatrixRXd& m2, BaseHash& hashes) {
    int n = m1.rows();
    int b = hashes.b;
    int d = hashes.d;

    MatrixRXcd p = MatrixRXcd::Zero(d, b);
    MatrixRXd p_real = MatrixRXd::Zero(d, b);

    int t, k, i, index1, index2;
    Complex d1, d2;

    Eigen::VectorXd pa = Eigen::VectorXd::Zero(b);
    Eigen::VectorXd pb = Eigen::VectorXd::Zero(b);

    // 🅱️alloc
    fftw_complex* out1 = fftw_alloc_complex(b / 2 + 1);
    fftw_complex* out2 = fftw_alloc_complex(b / 2 + 1);
    fftw_complex* in = fftw_alloc_complex(b);
    double* out = fftw_alloc_real(2 * (b / 2 + 1));

    fftw_plan plan1 = fftw_plan_dft_r2c_1d(b, pa.data(), out1, FFTW_ESTIMATE);
    fftw_plan plan2 = fftw_plan_dft_r2c_1d(b, pb.data(), out2, FFTW_ESTIMATE);

    for (t = 0; t < d; t++) {
        for (k = 0; k < n; k++) {
            pa.setZero();
            pb.setZero();

            for (i = 0; i < n; i++) {
                pa(hashes.hash(0, t, i)) += hashes.hash(2, t, i) * m1(i, k);
                pb(hashes.hash(1, t, i)) += hashes.hash(3, t, i) * m2(k, i);
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

    plan1 = fftw_plan_dft_c2r_1d(b, in, out, FFTW_ESTIMATE);

    for (t = 0; t < d; t++) {
        // Copy data from p to in, otherwise p would be overwritten later
        for (int i = 0; i < b; i++) {
            in[i][0] = p(t, i).real();
            in[i][1] = p(t, i).imag();
        }

        fftw_execute(plan1);

        for (i = 0; i < b; i++) {
            p_real(t, i) = out[i] / b;
        }
    }

    // Destroy the plans
    fftw_destroy_plan(plan1);
    fftw_destroy_plan(plan2);

    // Free EVERYTHING
    fftw_free(in);
    fftw_free(out1);
    fftw_free(out2);
    fftw_free(out);

    return p_real;
}

MatrixRXd compressed_product_par(const MatrixRXd& m1, const MatrixRXd& m2, BaseHash& hashes) {
    int n = m1.rows();
    int b = hashes.b;
    int d = hashes.d;

    MatrixRXcd p = MatrixRXcd::Zero(d, b);

    // 🅱️alloc

    MatrixRXd pas = MatrixRXd::Zero(d, b);
    MatrixRXd pbs = MatrixRXd::Zero(d, b);

    fftw_complex* out1 = fftw_alloc_complex(d * (b / 2 + 1));
    fftw_complex* out2 = fftw_alloc_complex(d * (b / 2 + 1));

    std::vector<fftw_plan> plans1;
    std::vector<fftw_plan> plans2;

    for (int t = 0; t < d; t++) {
        int offset = t * b / 2 + 1;
        plans1.push_back(fftw_plan_dft_r2c_1d(b, pas.row(t).data(), out1 + offset, FFTW_ESTIMATE));
        plans2.push_back(fftw_plan_dft_r2c_1d(b, pbs.row(t).data(), out2 + offset, FFTW_ESTIMATE));
    }

#pragma omp parallel for
    for (int t = 0; t < d; t++) {
        Complex d1;
        Complex d2;
        int offset = t * b / 2 + 1;
        for (int k = 0; k < n; k++) {
            pas.row(t).setZero();
            pbs.row(t).setZero();
            for (int i = 0; i < n; i++) {
                pas(t, hashes.hash(0, t, i)) += hashes.hash(2, t, i) * m1(i, k);
                pbs(t, hashes.hash(1, t, i)) += hashes.hash(3, t, i) * m2(k, i);
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
    for (int i = 0; i < n*n; i++) {
        int row = i / n;
        int col = i % n;
        c(row, col) = decompress_element(p, row, col, d, b, hashes);
    }
    return c;
}
