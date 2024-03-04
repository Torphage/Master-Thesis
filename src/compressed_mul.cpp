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

    fftw_plan plan = fftw_plan_dft_c2r_1d(b, in, out, FFTW_MEASURE); // Used to be FFTW_ESTIMATE
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
    MatrixRXd result = MatrixRXd::Zero(d, b);

    fftw_complex* in_size = reinterpret_cast<fftw_complex*>(p.row(0).data());
    double* out = fftw_alloc_real(b * sizeof(double));

    fftw_plan plan = fftw_plan_dft_c2r_1d(b, in_size, out, FFTW_MEASURE | FFTW_PRESERVE_INPUT);

#pragma omp parallel for
    for (int t = 0; t < d; t++) {
        fftw_complex* in = reinterpret_cast<fftw_complex*>(p.row(t).data());
        fftw_execute_dft_c2r(plan, in, result.row(t).data());
    }

    result /= b;

    fftw_destroy_plan(plan);

    return result;
}

