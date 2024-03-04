#pragma once
#ifndef FFT_HPP
#define FFT_HPP

#include "utils.hpp"

#ifdef USE_MKL

struct fft_struct {};

fft_struct init_fft_mkl();
ifft_struct init_ifft_mkl();
void fft_mkl(fft_struct info, int in_offset, int out_offset);
void ifft_mkl(ifft_struct info, int in_offset, int out_offset);
void clean_fft_mkl(fft_struct info);
void clean_ifft_mkl(ifft_struct info);

#else

#include <fftw3.h>

/**
 * @brief A FFT struct containing the information that
 * fftw3 needs to run correctly.
 */
struct fft_struct {
    double* in;
    fftw_complex* out;
    fftw_plan plan;
};

struct ifft_struct {
    fftw_complex* in;
    double* out;
    fftw_plan plan;
};

fft_struct init_fft_fftw(int b, double* in, Complex* out);
ifft_struct init_ifft_fftw(int b, Complex* in, double* out);
void fft_fftw(fft_struct info, int in_offset, int out_offset);
void ifft_fftw(ifft_struct info, int in_offset, int out_offset);
void clean_fft_fftw(fft_struct info);
void clean_ifft_fftw(ifft_struct info);

#endif

fft_struct init_fft(int b, double* in, Complex *out);
ifft_struct init_ifft(int b, Complex* in, double* out);

void fft(fft_struct info, int in_offset, int out_offset);

void ifft(ifft_struct info, int in_offset, int out_offset);

void clean_fft(fft_struct info);
void clean_ifft(ifft_struct info);

#endif