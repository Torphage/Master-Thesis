#include "function.hpp"

#include <Eigen/Dense>
#include <random>
#include <vector>

int main() {
    int b = 5000, d = 13, n = 2000, m = 2000;

    unsigned int seed = 5;
    // unsigned int seed = std::random_device{}();
    std::mt19937_64 rng(seed);

    MatrixRXd a = MatrixRXd::Zero(n, m);

    // FullyRandomHash<uint64_t> hash(n, b, d, seed);
    MultiplyShiftHash<uint32_t, uint16_t> hash(d, seed);

    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            a(i, j) = distribution(rng);
        }
    }

    Eigen::ArrayXd x_bar = a.rowwise().mean();

    MatrixRXd m1 = a.colwise() - x_bar;
    MatrixRXd m2 = (a.colwise() - x_bar).matrix().transpose().array();

    int N = std::max(n, m);
    int k = std::min(n, m);
    m1.conservativeResize(N, N);
    m2.conservativeResize(N, N);


    // MatrixRXd real_result = (1.0 / (m - 1.0)) * (m1.matrix() * m2.matrix());
    // std::cout << "real_result: \n" << real_result.block(0,0,k,k) << std::endl << std::endl;

    MatrixRXd compressed = function::compress_dark(m1, m1.matrix().transpose().array(), N, b, d, hash);
    MatrixRXd result = function::decompress_dark(compressed, N, b, d, hash);
    result /= (m - 1.0);

    int count = 0;
    MatrixRXd new_result(k, k);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            if (abs(result(i, j)) > 0.05) {
                new_result(i, j) = result(i, j); 
                count++;
            }

        }
    }

    // std::cout << "result: \n" << new_result << std::endl;
    std::cout << "Number of non-zero elements in result: " << count << " of " << k*k << " possible (" << 100.0*count/(double(k*k)) << "%)" << std::endl;

    return 0;
}
