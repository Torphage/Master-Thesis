#include "compressed_mul.hpp"
#include "hashing.hpp"
#include "utils.hpp"

#include <iostream>
#include <random>

int main() {
    int b = 50, d = 5, n = 10;

    unsigned int seed = 5; // std::random_device{}();
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> uni(-1.0, 1.0);

    MatrixRXd m1 = sparse_matrix_generator(n, 0.1, rng);
    MatrixRXd m2 = sparse_matrix_generator(n, 0.1, rng);

    // MatrixRXd m1 = MatrixRXd::NullaryExpr(n,n,[&](){return uni(rng);});
    // MatrixRXd m2 = MatrixRXd::NullaryExpr(n,n,[&](){return uni(rng);});

    // FullyRandomHash<uint64_t> hash(n, b, d, seed);
    // MultiplyShiftHash<uint32_t, uint16_t> hash(d, seed);
    TabulationHash<uint32_t, uint32_t, 8> hash(d, seed);

    int thread_count = std::max(d, omp_get_max_threads());
    MatrixRXd compressed = MatrixRXd::Zero(d, b);
    MatrixRXd pas = MatrixRXd::Zero(thread_count, b);
    MatrixRXd pbs = MatrixRXd::Zero(thread_count, b);
    MatrixRXcd ps = MatrixRXcd::Zero(d, b / 2 + 1);
    MatrixRXcd out1(n * d, b / 2 + 1);
    MatrixRXcd out2(n * d, b / 2 + 1);
    fft_struct fft1 = init_fft(b, pas.data(), out1.data());
    fft_struct fft2 = init_fft(b, pbs.data(), out2.data());
    ifft_struct ifft = init_ifft(b, ps.data(), compressed.data());

    MatrixRXd result = MatrixRXd::Zero(n, n);
    MatrixRXd xt = MatrixRXd::Zero(n, d);

    bompressed_product_par_large_threaded_better(m1, m2, b, d, hash, compressed, pas, pbs, ps, out1, out2, fft1, fft2, ifft);

    debompress_matrix_par(compressed, n, b, d, hash, result, xt);

    clean_fft(fft1);
    clean_fft(fft2);
    clean_ifft(ifft);

    MatrixRXd real_product = m1 * m2;

    std::cout << "\n--------- Real result ---------" << std::endl;
    std::cout << "Sum of elements: " << sum_matrix(real_product) << std::endl;
    std::cout << real_product << std::endl;

    std::cout << "\n--------- Approximate result ---------" << std::endl;
    std::cout << "Sum of elements: " << sum_matrix(result) << std::endl;
    round_matrix(result, 12);
    std::cout << result << std::endl;

    return 0;
}