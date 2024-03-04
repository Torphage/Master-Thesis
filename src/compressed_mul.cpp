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

MatrixRXd compressed_ifft(MatrixRXcd& p, int b, int d) {
    MatrixRXd result = MatrixRXd::Zero(d, b);
    ifft_struct info = init_ifft(b, p.data(), result.data());

    for (int t = 0; t < d; t++) {
        ifft(info, t * b, t * b);
    }

    result /= b;
    clean_ifft(info);

    return result;
}

MatrixRXd compressed_ifft_par(MatrixRXcd& p, int b, int d) {
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
