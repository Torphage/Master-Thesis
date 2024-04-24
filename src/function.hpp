#pragma once
#ifndef FUNCTION_HPP_
#define FUNCTION_HPP_

#include "compressed_mul.hpp"
#include "utils.hpp"

#include <functional>

namespace function {

template <typename T>
MatrixRXd compress_seq(const MatrixRXd& m1, const MatrixRXd& m2, int n, int b, int d, T& hash) {
    MatrixRXd compressed = MatrixRXd::Zero(d, b);
    ArrayRXd pa = ArrayRXd::Zero(b);
    MatrixRXcd p = MatrixRXcd::Zero(d, b / 2 + 1);
    ArrayRXcd out1(b / 2 + 1);
    fft_struct fft1 = init_fft(b, pa.data(), out1.data());
    ifft_struct ifft1 = init_ifft(b, p.data(), compressed.data());

    bompressed_product_seq(m1, m2, n, b, d, hash, compressed, pa, p, out1, fft1, ifft1);

    clean_fft(fft1);
    clean_ifft(ifft1);
    
    return compressed;
}

template <typename T>
MatrixRXd compress_par(const MatrixRXd& m1, const MatrixRXd& m2, int n, int b, int d, T& hash) {
    MatrixRXd compressed = MatrixRXd::Zero(d, b);
    MatrixRXd pas = MatrixRXd::Zero(d, b);
    MatrixRXd pbs = MatrixRXd::Zero(d, b);
    MatrixRXcd p = MatrixRXcd::Zero(d, b / 2 + 1);
    MatrixRXcd out1 = MatrixRXcd::Zero(d, b / 2 + 1);
    MatrixRXcd out2 = MatrixRXcd::Zero(d, b / 2 + 1);
    fft_struct fft1 = init_fft(b, pas.data(), out1.data());
    fft_struct fft2 = init_fft(b, pbs.data(), out2.data());
    ifft_struct ifft1 = init_ifft(b, p.data(), compressed.data());

    bompressed_product_par(m1, m2, n, b, d, hash, compressed, pas, pbs, p, out1, out2, fft1, fft2, ifft1);

    clean_fft(fft1);
    clean_fft(fft2);
    clean_ifft(ifft1);

    return compressed;
};

template <typename T>
MatrixRXd compress_threaded(const MatrixRXd& m1, const MatrixRXd& m2, int n, int b, int d, T& hash) {
    int threads = omp_get_max_threads();
    MatrixRXd compressed = MatrixRXd::Zero(d, b);
    MatrixRXd pas = MatrixRXd::Zero(threads, b);
    MatrixRXd pbs = MatrixRXd::Zero(threads, b);
    MatrixRXcd p = MatrixRXcd::Zero(d, b / 2 + 1);
    MatrixRXcd out1(threads, b / 2 + 1);
    MatrixRXcd out2(threads, b / 2 + 1);
    fft_struct fft1 = init_fft(b, pas.data(), out1.data());
    fft_struct fft2 = init_fft(b, pbs.data(), out2.data());
    ifft_struct ifft1 = init_ifft(b, p.data(), compressed.data());

    bompressed_product_par_threaded(m1, m2, n, b, d, hash, compressed, pas, pbs, p, out1, out2, fft1, fft2, ifft1);

    clean_fft(fft1);
    clean_fft(fft2);
    clean_ifft(ifft1);

    return compressed;
}

template <typename T>
MatrixRXd compress_deluxe(const MatrixRXd& m1, const MatrixRXd& m2, int n, int b, int d, T& hash) {
    int threads = omp_get_max_threads();
    MatrixRXd compressed = MatrixRXd::Zero(d, b);
    MatrixRXd pas = MatrixRXd::Zero(threads, b);
    MatrixRXcd p = MatrixRXcd::Zero(d, b / 2 + 1);
    MatrixRXcd out1(threads, b / 2 + 1);
    MatrixRXcd out2(threads, b / 2 + 1);
    fft_struct fft1 = init_fft(b, pas.data(), out1.data());
    fft_struct fft2 = init_fft(b, pas.data(), out2.data());
    ifft_struct ifft1 = init_ifft(b, p.data(), compressed.data());

    bompressed_product_par_deluxe(m1, m2, n, b, d, hash, compressed, pas, p, out1, out2, fft1, fft2, ifft1);

    clean_fft(fft1);
    clean_fft(fft2);
    clean_ifft(ifft1);

    return compressed;
}

template <typename T>
MatrixRXd compress_secret(const MatrixRXd& m1, const MatrixRXd& m2, int n, int b, int d, T& hash) {
    int threads = omp_get_max_threads();
    int size = std::max(2 * threads, d);

    MatrixRXd pas = MatrixRXd::Zero(size, b);
    MatrixRXcd p = MatrixRXcd::Zero(d, b / 2 + 1);
    MatrixRXcd out(4 * threads, b / 2 + 1);
    fft_struct fft1 = init_fft(b, pas.data(), out.data());
    ifft_struct ifft1 = init_ifft(b, p.data(), pas.data());

    bompressed_product_par_secret_dark_tech_edition(m1, m2, n, b, d, hash, pas, p, out, fft1, ifft1);

    clean_fft(fft1);
    clean_ifft(ifft1);

    return pas;
}

// template <typename T>
// void compress_special(const MatrixRXd& m1, const MatrixRXd& m2, int n, int b, int d, T& hash) {
//     int threads = omp_get_max_threads();
//     MatrixRXd compressed = MatrixRXd::Zero(d, b);
//     MatrixRXd pas = MatrixRXd::Zero(threads, b);
//     MatrixRXcd p = MatrixRXcd::Zero(d, b / 2 + 1);
//     MatrixRXcd out1(threads, b / 2 + 1);
//     MatrixRXcd out2(threads, b / 2 + 1);
//     fft_struct fft1 = init_fft(b, pas.data(), out1.data());
//     fft_struct fft2 = init_fft(b, pas.data(), out2.data());
//     ifft_struct ifft1 = init_ifft(b, p.data(), pas.data());

//     return [=]() mutable {
//         bompressed_product_par_special(m1, m2, n, b, d, hash, compressed, pas, p, out1, out2, fft1, fft2, ifft1);
//         return compressed; };
// }

// template <typename T>
// void decompress_seq();

// template <typename T>
// class decompress_par {
//    private:
//     MatrixRXd compressed;
//     int n, b, d;
//     T hash;
//     MatrixRXd result;
//     MatrixRXd xt;

//    public:
//     MatrixRXd operator()(void) {
//         debompress_matrix_par(compressed, n, b, d, hash, result, xt);
//         return result;
//     };
// };

template <typename T>
MatrixRXd decompress_par(MatrixRXd& compressed, int n, int b, int d, T& hash) {
    MatrixRXd result = MatrixRXd::Zero(n, n);
    MatrixRXd xt = MatrixRXd::Zero(n, d);

    debompress_matrix_par(compressed, n, b, d, hash, result, xt);
    return result;
}

template <typename T>
MatrixRXd decompress_threaded(MatrixRXd& compressed, int n, int b, int d, T& hash) {
    MatrixRXd result = MatrixRXd::Zero(n, n);
    MatrixRXd xt = MatrixRXd::Zero(n, d);

    debompress_matrix_par_threaded(compressed, n, b, d, hash, result, xt);
    return result;
}

}  // namespace function

#endif
