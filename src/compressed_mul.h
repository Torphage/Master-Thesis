#include <Eigen/Dense>
#include <complex>
#include <fftw3.h>


typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> complex_dyn_matrix;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> double_vector;
typedef Eigen::Matrix<fftw_complex, Eigen::Dynamic, 1> complex_vector;

struct params {
    const int b;
    const int d;
};

struct hashes {
    Eigen::MatrixXi h1;
    Eigen::MatrixXi h2;
    Eigen::MatrixXi s1;
    Eigen::MatrixXi s2;
};

Eigen::MatrixXcd compressed_product(const Eigen::MatrixXd &m1, const Eigen::MatrixXd &m2, const hashes &hs, const params &ps);

Eigen::MatrixXd decompress_matrix(Eigen::MatrixXcd p, hashes hs, params ps, int n);

double find_median(Eigen::VectorXd vec);