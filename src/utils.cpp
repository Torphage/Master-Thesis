#include "utils.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <random>

MatrixRXd sparse_matrix_generator(int n, float density, std::mt19937_64 &rng) {
    std::uniform_real_distribution<float> uni(-1.0, 1.0);
    MatrixRXd m = MatrixRXd::NullaryExpr(n, n, [&]() { return uni(rng); });
    ;

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

void round_matrix(MatrixRXd &matrix, int n) {
    for (int i = 0; i < matrix.rows(); i++) {
        for (int j = 0; j < matrix.cols(); j++) {
            if (abs(matrix(i, j)) < pow(0.1, n)) {
                matrix(i, j) = std::round(matrix(i, j) * n * 10.0) / (n * 10.0);
            }
        }
    }
}

double sum_matrix(MatrixRXd &matrix) {
    double res = 0;
    for (int i = 0; i < matrix.rows(); i++) {
        for (int j = 0; j < matrix.cols(); j++) {
            res += matrix(i, j);
        }
    }
    return res;
}

void progress_bar(double percentage) {
    int barWidth = 70;

    std::cout << "[";
    int pos = barWidth * percentage;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << int(percentage * 100.0) << " %\r";
    std::cout.flush();
}
