#include <iostream>
#include <random>
#include "compressed_mul.h"


int main() {
    int b = 5000, d = 30, n = 10;
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_real_distribution<float> dis(-1.0, 1.0);
    
    Eigen::MatrixXi h1(d, n), h2(d, n), s1(d, n), s2(d, n);
    std::uniform_int_distribution<int> dist(0, b - 1);

    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < n; ++j) {
            h1(i, j) = dist(rng);
            h2(i, j) = dist(rng);
            s1(i, j) = (rng() % 2 == 0) ? 1 : -1;
            s2(i, j) = (rng() % 2 == 0) ? 1 : -1;
        }
    }

    struct hashes hs = {h1, h2, s1, s2};
    struct params ps = {b, d};
    
    
    Eigen::VectorXd test = Eigen::VectorXd::NullaryExpr(5,[&](){return dis(rng);});

    std::cout << "Vector:\n" << test << std::endl;
    std::cout << "Median:\n" << find_median(test) << std::endl;


    Eigen::MatrixXd m1 = Eigen::MatrixXd::NullaryExpr(n,n,[&](){return dis(rng);});
    Eigen::MatrixXd m2 = Eigen::MatrixXd::NullaryExpr(n,n,[&](){return dis(rng);});

    Eigen::MatrixXcd p = compressed_product(m1, m2, hs, ps);
    Eigen::MatrixXd result = decompress_matrix(p, hs, ps, n);

    std::cout << "REAL Result:\n" << m1*m2 << std::endl;
    
    std::cout << "Result:\n" << result << std::endl;

    return 0;
}