#include <iostream>
#include <random>
#include "compressed_mul.hpp"
#include "hashing.hpp"
#include "utils.hpp"


int main() {
    int b = 40, d = 17, n = 16;

    unsigned int seed = std::random_device{}();
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> uni(-1.0, 1.0);

    Eigen::MatrixXd m1 = sparse_matrix_generator(n, 0.05, rng);
    Eigen::MatrixXd m2 = sparse_matrix_generator(n, 0.05, rng);

    // Eigen::MatrixXd m1 = Eigen::MatrixXd::NullaryExpr(n,n,[&](){return uni(rng);});
    // Eigen::MatrixXd m2 = Eigen::MatrixXd::NullaryExpr(n,n,[&](){return uni(rng);});

    // FullyRandomHash hashes(n, b, d, rng);
    // MultiplyShiftHash hashes(b, d, rng);
    int p = 32, q = 32, r = 8;
    TabulationHash hashes(p, q, r, b ,d, rng);
    

    Eigen::MatrixXd prod = compressed_product(m1, m2, hashes);
    Eigen::MatrixXd result = decompress_matrix(prod, n, hashes);

    Eigen::MatrixXd real_product = m1*m2;

    std::cout << "\n--------- Real result ---------" << std::endl;
    std::cout << "Sum of elements: " << sum_matrix(real_product) << std::endl;
    std::cout << real_product << std::endl;
    
    std::cout << "\n--------- Approximate result ---------" << std::endl;
    std::cout << "Sum of elements: " << sum_matrix(result) << std::endl;
    round_matrix(result, 12);
    std::cout << result << std::endl;

    return 0;
}