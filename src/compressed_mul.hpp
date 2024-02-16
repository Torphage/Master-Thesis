#include <Eigen/Dense>
#include <complex>
#include <fftw3.h>


typedef std::complex<double> Complex;

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

Eigen::MatrixXd compressed_product(const Eigen::MatrixXd &m1, const Eigen::MatrixXd &m2, const hashes &hs, const params &ps);

Eigen::MatrixXd decompress_matrix(Eigen::MatrixXd p, hashes hs, params ps, int n);

double find_median(Eigen::VectorXd vec);