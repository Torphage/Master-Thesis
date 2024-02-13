#include <iostream>
#include <random>
#include "compressed_mul.h"


int main() {
    int b = 5, d = 2, n = 10;

    Eigen::MatrixXi h1(d, n), h2(d, n), s1(d, n), s2(d, n);
    unsigned int seed = std::random_device{}();
    std::mt19937_64 rng(123);
    std::uniform_real_distribution<float> uni(-1.0, 1.0);
    std::uniform_int_distribution<int> h_hash(0, b - 1);

    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < n; ++j) {
            h1(i, j) = h_hash(rng);
            h2(i, j) = h_hash(rng);
            s1(i, j) = (rng() % 2 == 0) ? 1 : -1;
            s2(i, j) = (rng() % 2 == 0) ? 1 : -1;
        }
    }

    struct hashes hs = {h1, h2, s1, s2};
    struct params ps = {b, d};
    
    Eigen::MatrixXd m1 = Eigen::MatrixXd::NullaryExpr(n,n,[&](){return uni(rng);});
    Eigen::MatrixXd m2 = Eigen::MatrixXd::NullaryExpr(n,n,[&](){return uni(rng);});

    Eigen::MatrixXd p = compressed_product(m1, m2, hs, ps);
    Eigen::MatrixXd result = decompress_matrix(p, hs, ps, n);

    std::cout << "REAL Result:\n" << m1*m2 << std::endl;
    
    std::cout << "Result:\n" << result << std::endl;

    return 0;
}