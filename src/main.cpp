#include <iostream>
#include <random>

#include "compressed_mul.hpp"
#include "hashing.hpp"
#include "utils.hpp"

int main() {
    int b = 15, d = 14, n = 10;

    unsigned int seed = std::random_device{}();
    std::mt19937_64 rng(seed);
    std::mt19937_64 rng2(2);
    std::uniform_real_distribution<float> uni(-1.0, 1.0);

    MatrixRXd m1 = sparse_matrix_generator(n, 0.1, rng);
    MatrixRXd m2 = sparse_matrix_generator(n, 0.1, rng);

    // MatrixRXd m1 = MatrixRXd::NullaryExpr(n,n,[&](){return uni(rng);});
    // MatrixRXd m2 = MatrixRXd::NullaryExpr(n,n,[&](){return uni(rng);});

    Hashes<Eigen::MatrixXi> hashes = fully_random_constructor(n, b, d, rng);
    // Hashes<MatrixXui> hashes = multiply_shift_constructor(d, rng);
    // int p = 32, q = 32, r = 8;
    // Hashes<std::vector<MatrixXui>> hashes = tabulation_constructor(p, q, r, d, rng);

    MatrixRXd prod = compressed_product(m1, m2, b, d, fully_random_hash(), hashes);
    MatrixRXd result = decompress_matrix(prod, n, b, d, fully_random_hash(), hashes);

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