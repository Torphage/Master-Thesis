#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <memory>
#include <omp.h>

#include "compressed_mul.hpp"
#include "utils.hpp"
#include "hashing.hpp"
#include "variance.hpp"

double variance(Eigen::VectorXd &vec) {
    Eigen::VectorXd temp = vec.array() - vec.mean();
    Eigen::VectorXd ab = temp.array().abs();
    Eigen::VectorXd val = ab.array().pow(2);
    return val.sum() / (val.size() - 1);
}

MatrixRXd variance3d(std::vector<MatrixRXd> &mat) {
    int n = mat[0].cols(), m = mat[0].rows();
    MatrixRXd result = MatrixRXd::Zero(n, m);
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

bool test_variance(MatrixRXd m1, MatrixRXd m2, HashInfo &h, int num_samples) {    
    int n = m1.rows();
    int b = h.b;
    int d = h.d;

    MatrixRXd compressed;
    MatrixRXd decompressed;

    std::vector<MatrixRXd> vec;
    std::unique_ptr<BaseHash> hashes;

    for (int i = 0; i < num_samples; i++) {
        if (h.id == "FullyRandomHash") {
            hashes = std::make_unique<FullyRandomHash>(h.n, h.b, h.d, h.rng);
        } else if (h.id == "MultiplyShiftHash") {
            hashes = std::make_unique<MultiplyShiftHash>(h.b, h.d, h.rng);
        } else if (h.id == "TabulationHash") {
            hashes = std::make_unique<TabulationHash>(h.p, h.q, h.r, h.b, h.d, h.rng);
        }

        compressed = compressed_product(m1, m2, *hashes);
        decompressed = decompress_matrix(compressed, n, *hashes);

        vec.push_back(decompressed);
        if ((i + 1) % (num_samples / 100) == 0) {
            progress_bar((i + 1.0) / (num_samples));
        }
    }
    MatrixRXd result = variance3d(vec);

    // std::cout << "Variance:" << "\n" << result << std::endl << std::endl;
    // std::cout << "Sum of variance: " << sum_matrix(result) << std::endl;
    double bound =  (pow((m1*m2).norm(), 2)) / b;
    // std::cout << "Bound: " << bound << std::endl;
    return (result.array() < bound).all();
}

