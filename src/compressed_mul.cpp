#include <cassert>
#include <cmath>
#include <complex>
#include <fftw3.h>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>
#include "compressed_mul.hpp"


Eigen::MatrixXd compressed_product(const Eigen::MatrixXd &m1, const Eigen::MatrixXd &m2, BaseHash &hashes) {
    int n = m1.rows();
    int b = hashes.b;
    int d = hashes.d;

    Eigen::MatrixXcd p = Eigen::MatrixXcd::Zero(d, b);
    Eigen::MatrixXd p_real = Eigen::MatrixXd::Zero(d, b);

    int t, k, i, index1, index2;

    Eigen::VectorXd pa, pb;
    Complex d1, d2;
    fftw_plan plan1, plan2;

    // 🅱️alloc
    fftw_complex* out1 = fftw_alloc_complex(b/2+1);
    fftw_complex* out2 = fftw_alloc_complex(b/2+1);
    fftw_complex* in = fftw_alloc_complex(b);
    double* out = fftw_alloc_real(2*(b/2+1));

    for (t = 0; t < d; t++) {
        for (k = 0; k < n; k++) {
            pa = Eigen::VectorXd::Zero(b);
            pb = Eigen::VectorXd::Zero(b);
            
            for (i = 0; i < n; i++) {
                pa(hashes.hash("h1", t, i)) += hashes.hash("s1", t, i) * m1(i, k);
                pb(hashes.hash("h2", t, i)) += hashes.hash("s2", t, i) * m2(k, i);
            }

            plan1 = fftw_plan_dft_r2c_1d(b, pa.data(), out1, FFTW_ESTIMATE);
            plan2 = fftw_plan_dft_r2c_1d(b, pb.data(), out2, FFTW_ESTIMATE);

            fftw_execute(plan1);
            fftw_execute(plan2);

        
            for (i = 0; i < b/2 + 1; i++) {
                d1 = Complex(out1[i][0], out1[i][1]);
                d2 = Complex(out2[i][0], out2[i][1]);
                p(t, i) += d1 * d2;
            }        

        } 
    }

    for (t = 0; t < d; t++) {
        // Copy data from p to in, otherwise p would be overwritten later
        for (int i = 0; i < b; i++) {
            in[i][0] = p(t, i).real();
            in[i][1] = p(t, i).imag();
        }

        plan1 = fftw_plan_dft_c2r_1d(b, in, out, FFTW_ESTIMATE);

        fftw_execute(plan1);

        for (i = 0; i < b; i++) {
            p_real(t, i) = out[i] / b;
        }

    }

    // Destroy the plans
    fftw_destroy_plan(plan1);
    fftw_destroy_plan(plan2);

    // Free EVERYTHING
    fftw_free(in);
    fftw_free(out1);
    fftw_free(out2);
    fftw_free(out);

    return p_real;

}


double find_median(Eigen::VectorXd vec) {
    int n = vec.size();
    int targetIndex = n / 2;
    double *data = vec.data();
    
    std::nth_element(data, data + n / 2, data + n); 
    double median1 = vec(targetIndex);

    if (n % 2 != 0) { 
        return median1; 
    } 

    std::nth_element(data, data + (n - 1) / 2, data + n); 
    double median2 = vec(targetIndex - 1);
  
    return (median1 + median2) / 2.0; 
} 


double decompress_element(const Eigen::MatrixXd &p, int i, int j, int d, int b, BaseHash &hashes) {
    Eigen::VectorXd xt = Eigen::VectorXd::Zero(d);
    for (int t = 0; t < d; t++) {
        xt(t) = hashes.hash("s1", t, i) * hashes.hash("s2", t, j) * 
        p(t, (hashes.hash("h1", t, i) + hashes.hash("h2", t, j)) % b);
    } 

    return find_median(xt);
}

Eigen::MatrixXd decompress_matrix(Eigen::MatrixXd p, int n, BaseHash &hashes) {
    int b = hashes.b;
    int d = hashes.d;
    Eigen::MatrixXd c = Eigen::MatrixXd::Zero(n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            c(i, j) = decompress_element(p, i, j, d, b, hashes);
        }
    }
    return c;
}