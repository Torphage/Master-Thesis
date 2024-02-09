#include <iostream>
#include <random>
#include "compressed_mul.h"


int main() {
    int b = 10, d = 3, n = 100;
    std::mt19937_64 rng(std::random_device{}());
    
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
    
            
    Eigen::MatrixXd m1 = Eigen::MatrixXd::Random(n, n);
    Eigen::MatrixXd m2 = Eigen::MatrixXd::Random(n, n);

    Eigen::MatrixXcd p = compressed_product(m1, m2, hs, ps);

    std::cout << "Result:\n" << p << std::endl;

    return 0;
}