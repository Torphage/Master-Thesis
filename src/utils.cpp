#include <Eigen/Dense>
#include <random>
#include <cmath>



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

void round_matrix(Eigen::MatrixXd &matrix, int n) {
    for (int i = 0; i < matrix.rows(); i++) {
        for (int j = 0; j < matrix.cols(); j++) {
            if (abs(matrix(i,j)) < pow(0.1, n)) {
                matrix(i, j) = std::round(matrix(i, j) * n * 10.0) / (n * 10.0);
            } 
        }
    }
}

double sum_matrix(Eigen::MatrixXd &matrix) {
    double res = 0;
    for (int i = 0; i < matrix.rows(); i++) {
        for (int j = 0; j < matrix.cols(); j++) {
            res += matrix(i, j);
        }
    }
    return res;
}

