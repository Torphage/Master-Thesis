#include "compressed_mul.hpp"
#include "hashing.hpp"
#include "utils.hpp"

#include <iostream>
#include <random>

int main() {
    int b = 15, d = 14, n = 10;

    unsigned int seed = std::random_device{}();
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> uni(-1.0, 1.0);

    MatrixRXd m1 = sparse_matrix_generator(n, 0.1, rng);
    MatrixRXd m2 = sparse_matrix_generator(n, 0.1, rng);

    // MatrixRXd m1 = MatrixRXd::NullaryExpr(n,n,[&](){return uni(rng);});
    // MatrixRXd m2 = MatrixRXd::NullaryExpr(n,n,[&](){return uni(rng);});

    // FullyRandomHash hash(n, b, d, seed);
    // MultiplyShiftHash hash(d, seed);
    int p = 32, q = 32, r = 8;
    TabulationHash hash(p, q, r, d, seed);
    
    MatrixRXd compressed = MatrixRXd::Zero(d, b);
    MatrixRXd pas = MatrixRXd::Zero(d, b);
    MatrixRXd pbs = MatrixRXd::Zero(d, b);
    MatrixRXd result = MatrixRXd::Zero(n, n);
    MatrixRXd xt = MatrixRXd::Zero(n, d);

    bompressed_product_par(m1, m2, b, d, hash, compressed, pas, pbs);
    debompress_matrix_par(compressed, n, b, d, hash, result, xt);

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