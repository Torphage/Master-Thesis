#include "variance.hpp"

#include "compressed_mul.hpp"
#include "hashing.hpp"
#include "utils.hpp"

#include <omp.h>

#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <vector>

double variance(Eigen::VectorXd &vec) {
    Eigen::VectorXd temp = vec.array() - vec.mean();
    temp = temp.array().abs();
    temp = temp.array().pow(2);
    return temp.sum() / (temp.size() - 1);
}

MatrixRXd variance3d(std::vector<MatrixRXd> &mat) {
    int n = mat[0].cols(), m = mat[0].rows();
    MatrixRXd result = MatrixRXd::Zero(n, m);
    for (int j = 0; j < n; j++) {      // The row of 2d matrix
        for (int k = 0; k < m; k++) {  // The column of 2d matrix
            Eigen::VectorXd vec = Eigen::VectorXd::Zero(mat.size());
            for (unsigned long i = 0; i < mat.size(); i++) {  // Which matrix
                vec(i) = mat[i](j, k);
            }
            result(j, k) = variance(vec);
        }
    }
    return result;
}
