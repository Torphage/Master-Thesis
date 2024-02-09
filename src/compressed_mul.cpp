#include <iostream>
#include <random>
#include <tuple>
#include <complex>
#include <fftw3.h>
#include "compressed_mul.h"
#include <vector>
#include <cmath>

typedef std::complex<double> Complex;


Eigen::MatrixXcd compressed_product(const Eigen::MatrixXd &m1, const Eigen::MatrixXd &m2, const hashes &hs, const params &ps) {
    
    int n = m1.rows();
    
    Eigen::MatrixXcd p = Eigen::MatrixXcd::Zero(ps.d, ps.b);

    int t, k, i, z;
   

    for (t = 0; t < ps.d; t++) {
        for (k = 0; k < n; k++) {
            Eigen::VectorXcd pa = Eigen::VectorXcd::Zero(ps.b);
            Eigen::VectorXcd pb = Eigen::VectorXcd::Zero(ps.b);
            

            for (i = 0; i < n; i++) {
                pa(hs.h1(t, i)) += hs.s1(t, i) * m1(i, k);
                pb(hs.h2(t, i)) += hs.s2(t, i) * m2(k, i);
            }

            fftw_complex* in1 = reinterpret_cast<fftw_complex*>(pa.data());
            fftw_complex* in2 = reinterpret_cast<fftw_complex*>(pb.data());
            fftw_complex* out1 = reinterpret_cast<fftw_complex*>(pa.data());
            fftw_complex* out2 = reinterpret_cast<fftw_complex*>(pb.data());

            fftw_plan plan1 = fftw_plan_dft_1d(ps.b, in1, out1, FFTW_FORWARD, FFTW_ESTIMATE);
            fftw_plan plan2 = fftw_plan_dft_1d(ps.b, in2, out2, FFTW_FORWARD, FFTW_ESTIMATE);

            fftw_execute(plan1);
            fftw_execute(plan2);
        
            fftw_destroy_plan(plan1);
            fftw_destroy_plan(plan2);

            for (z = 0; z < ps.b; z++) {
                p(t, z) += pa(z) * pb(z); 
            }
        } 
    }

    for (t = 0; t < ps.d; t++) {
        fftw_complex* in = reinterpret_cast<fftw_complex*>(p.row(t).data());
        fftw_complex* out = reinterpret_cast<fftw_complex*>(p.row(t).data());
        fftw_plan plan = fftw_plan_dft_1d(ps.b, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);

        fftw_execute(plan);
        fftw_destroy_plan(plan);
    }


    // Normalize since fftw doesn't >:(
    for (int i = 0; i < ps.d; i++) {
        for (int j = 0; j < ps.b; j++) {
            p(i, j) /= ps.b;
        }
    }

    return p;
}

int decompress_element(int i, int j, Eigen::MatrixXd p, hashes hs, params ps) {
    return 0;
}