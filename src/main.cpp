#include <iostream>
#include <random>
#include "compressed_mul.hpp"


/**
 * @brief Generates a random sparse square matrix
 * 
 * @param n is the width and height of the generated matrix
 * @param density is the wanted sparsity, specifically the percentage
 *                of how many non-zeros that should be generated 
 * @param rng is the random number generator
 * @return A random square sparse matrix
 */
Eigen::MatrixXd sparse_matrix_generator(int n, float density, std::mt19937_64 &rng) {
    std::uniform_real_distribution<float> uni(-1.0, 1.0);
    Eigen::MatrixXd m = Eigen::MatrixXd::NullaryExpr(n,n,[&](){return uni(rng);});;

    int num_zeros = static_cast<int>(n * n * (1 - density));
    std::vector<int> indices(n * n);
    for (int i = 0; i < n * n; i++) {
        indices[i] = i;
    }

    std::shuffle(indices.begin(), indices.end(), rng);
    for (int i = 0; i < num_zeros; i++) {
        int index = indices[i];
        m(index / n, index % n) = 0.0;
    }

    return m;
}

int main() {
    int b = 15, d = 16, n = 15;

    unsigned int seed = std::random_device{}();
    std::mt19937_64 rng(123);
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

    std::cout << "REAL Result:\n" << m1*m2 << std::endl;
    
    std::cout << "Result:\n" << result << std::endl;

    return 0;
}