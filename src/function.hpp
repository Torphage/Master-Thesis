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
    Eigen::Array<Complex, 2, Eigen::Dynamic, Eigen::RowMajor> out1(2, b / 2 + 1);
    fft::fft_struct fft1 = fft::init_fft_struct(b, pa.data(), out1.data());
    fft::ifft_struct ifft1 = fft::init_ifft_struct(b, p.data(), compressed.data());

    MatrixRXd m1t = m1.matrix().transpose().array();

    bompressed_product_seq(m1t, m2, n, b, d, hash, compressed, pa, p, out1, fft1, ifft1);

    fft::clean_fft(fft1);
    fft::clean_ifft(ifft1);

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
    fft::fft_struct fft1 = fft::init_fft_struct(b, pas.data(), out1.data());
    fft::fft_struct fft2 = fft::init_fft_struct(b, pbs.data(), out2.data());
    fft::ifft_struct ifft1 = fft::init_ifft_struct(b, p.data(), compressed.data());

    MatrixRXd m1t = m1.matrix().transpose().array();

    bompressed_product_par(m1t, m2, n, b, d, hash, compressed, pas, pbs, p, out1, out2, fft1, fft2, ifft1);

    fft::clean_fft(fft1);
    fft::clean_fft(fft2);
    fft::clean_ifft(ifft1);

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
    fft::fft_struct fft1 = fft::init_fft_struct(b, pas.data(), out1.data());
    fft::fft_struct fft2 = fft::init_fft_struct(b, pbs.data(), out2.data());
    fft::ifft_struct ifft1 = fft::init_ifft_struct(b, p.data(), compressed.data());

    MatrixRXd m1t = m1.matrix().transpose().array();

    bompressed_product_par_threaded(m1t, m2, n, b, d, hash, compressed, pas, pbs, p, out1, out2, fft1, fft2, ifft1);

    fft::clean_fft(fft1);
    fft::clean_fft(fft2);
    fft::clean_ifft(ifft1);

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
    fft::fft_struct fft1 = fft::init_fft_struct(b, pas.data(), out1.data());
    fft::fft_struct fft2 = fft::init_fft_struct(b, pas.data(), out2.data());
    fft::ifft_struct ifft1 = fft::init_ifft_struct(b, p.data(), compressed.data());

    MatrixRXd m1t = m1.matrix().transpose().array();

    bompressed_product_par_deluxe(m1t, m2, n, b, d, hash, compressed, pas, p, out1, out2, fft1, fft2, ifft1);

    fft::clean_fft(fft1);
    fft::clean_fft(fft2);
    fft::clean_ifft(ifft1);

    return compressed;
}

template <typename T>
MatrixRXd compress_secret(const MatrixRXd& m1, const MatrixRXd& m2, int n, int b, int d, T& hash) {
    int threads = omp_get_max_threads();
    int size = std::max(2 * threads, d);

    MatrixRXd pas = MatrixRXd::Zero(size, b);
    MatrixRXcd p = MatrixRXcd::Zero(d, b / 2 + 1);
    MatrixRXcd out(4 * threads, b / 2 + 1);
    fft::fft_struct fft1 = fft::init_fft_struct(b, pas.data(), out.data());
    fft::ifft_struct ifft1 = fft::init_ifft_struct(b, p.data(), pas.data());

    MatrixRXd m1t = m1.matrix().transpose().array();

    bompressed_product_par_secret_dark_tech_edition(m1t, m2, n, b, d, hash, pas, p, out, fft1, ifft1);

    fft::clean_fft(fft1);
    fft::clean_ifft(ifft1);

    return pas;
}

template <typename T>
MatrixRXd compress_dark(const MatrixRXd& m1, const MatrixRXd& m2, int n, int b, int d, T& hash) {
    int b2 = b / 2 + 1;

    MatrixRXd compressed = MatrixRXd::Zero(d, b);
    MatrixRXcd p = MatrixRXcd::Zero(d, b2);
    MatrixRX2i hashes1 = MatrixRX2i::Zero(n, 2);
    MatrixRX2i hashes2 = MatrixRX2i::Zero(n, 2);
    MatrixRXcd out(d, b2);
    fft::fft_plan fft1 = fft::init_fft(b, compressed.data(), out.data());
    fft::fft_plan ifft1 = fft::init_ifft(b, p.data(), compressed.data());

    MatrixRXd m1t = m1.matrix().transpose().array();

    bompressed_product_par_dark(m1t, m2, n, b, d, hash, compressed, p, hashes1, hashes2, fft1, ifft1);

    fft::clean(fft1);
    fft::clean(ifft1);

    return compressed;
}

// template <typename T>
// void compress_special(const MatrixRXd& m1, const MatrixRXd& m2, int n, int b, int d, T& hash) {
//     int threads = omp_get_max_threads();
//     MatrixRXd compressed = MatrixRXd::Zero(d, b);
//     MatrixRXd pas = MatrixRXd::Zero(threads, b);
//     MatrixRXcd p = MatrixRXcd::Zero(d, b / 2 + 1);
//     MatrixRXcd out1(threads, b / 2 + 1);
//     MatrixRXcd out2(threads, b / 2 + 1);
//     fft::fft_struct fft1 = fft::init_fft_struct(b, pas.data(), out1.data());
//     fft::fft_struct fft2 = fft::init_fft_struct(b, pas.data(), out2.data());
//     fft::ifft_struct ifft1 = fft::init_ifft_struct(b, p.data(), pas.data());

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
    MatrixRXd xt = MatrixRXd::Zero(omp_get_max_threads(), d);

    debompress_matrix_par_threaded(compressed, n, b, d, hash, result, xt);
    return result;
}

template <typename T>
MatrixRXd decompress_dark(MatrixRXd& compressed, int n, int b, int d, T& hash) {
    MatrixRXd result(n, n);

    debompress_matrix_par_dark(compressed, n, b, d, hash, result);
    return result;
}

}  // namespace function

#endif
