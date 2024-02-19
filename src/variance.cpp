#include <Eigen/Dense>
#include <vector>
#include <iostream>

double variance(Eigen::VectorXd &vec) {
    Eigen::VectorXd temp = vec.array() - vec.mean();
    Eigen::VectorXd val = temp.array().pow(2);
    return val.sum() / (vec.norm() - 1);
}

Eigen::MatrixXd variance3d(std::vector<Eigen::MatrixXd> &mat) {
    int n = mat[0].rows(), m = mat[0].rows();
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(n, m);
    for (int j = 0; j < n; j++) { // The row of 2d matrix
        for (int k = 0; k < m; k++) { // The column of 2d matrix
            Eigen::VectorXd vec = Eigen::VectorXd::Zero(mat.size());
            for (int i = 0; i < mat.size(); i++) { // Which matrix
                vec(i) = mat[i](j, k);
            }
            result(j, k) = variance(vec);
        }
    }
    return result;
}