#include "compressed_mul.hpp"
#include "function.hpp"
#include "hashing.hpp"
#include "utils.hpp"

#include <iostream>
#include <random>

int main() {
    const int b = 50, d = 5, n = 10;

    unsigned int seed = 5;  // std::random_device{}();
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> uni(-1.0, 1.0);

    MatrixRXd m1 = sparse_matrix_generator(n, 0.1, rng);
    MatrixRXd m2 = sparse_matrix_generator(n, 0.1, rng);

    // std::cout << "m1: \n" << m1 << std::endl;
    // std::cout << "m2: \n" << m2 << std::endl;

    // MatrixRXd m1 = MatrixRXd::NullaryExpr(n,n,[&](){return uni(rng);});
    // MatrixRXd m2 = MatrixRXd::NullaryExpr(n,n,[&](){return uni(rng);});

    // FullyRandomHash<uint64_t> hash(n, b, d, seed);
    MultiplyShiftHash<uint32_t, uint16_t> hash(d, seed);
    // TabulationHash<uint32_t, uint32_t, 8> hash(d, seed);
    
    MatrixRXd compressed = function::compress_dark(m1, m2, n, b, d, hash);
    MatrixRXd result = function::decompress_dark(compressed, n, b, d, hash);

    MatrixRXd real_product = m1.matrix() * m2.matrix();

    std::cout << "\n--------- Real result ---------" << std::endl;
    std::cout << "Sum of elements: " << sum_matrix(real_product) << std::endl;
    std::cout << real_product << std::endl;

    std::cout << "\n--------- Approximate result ---------" << std::endl;
    std::cout << "Sum of elements: " << sum_matrix(result) << std::endl;
    round_matrix(result, 12);
    std::cout << result << std::endl;

    return 0;
}
