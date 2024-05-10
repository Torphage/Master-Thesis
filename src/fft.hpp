#pragma once
#ifndef FFT_HPP
#define FFT_HPP

#include "utils.hpp"

namespace fft {

#ifdef USE_MKL

struct fft_struct {
    double* in;
    Complex* out;
    DFTI_DESCRIPTOR_HANDLE descriptor;
};

struct ifft_struct {
    Complex* in;
    double* out;
    DFTI_DESCRIPTOR_HANDLE descriptor;
};

fft_struct init_fft_struct_mkl(int b, double* in, Complex* out);
ifft_struct init_ifft_struct_mkl(int b, Complex* in, double* out);
void fft_mkl(fft_struct info, int in_offset, int out_offset);
void ifft_mkl(ifft_struct info, int in_offset, int out_offset);
void clean_fft_mkl(fft_struct info);
void clean_ifft_mkl(ifft_struct info);

typedef DFTI_DESCRIPTOR_HANDLE fft_plan;

fft_plan init_fft_mkl(int b);
fft_plan init_ifft_mkl(int b);
void execute_fft_mkl(fft_plan& plan, double* in, Complex* out);
void execute_ifft_mkl(fft_plan& plan, Complex* in, double* out);
void clean_mkl(fft_plan& plan);

#else

#include <fftw3.h>

typedef fftw_plan fft_plan;

/**
 * @brief A FFT struct containing the information that
 * r2c FFT needs to run correctly.
 */
struct fft_struct {
    fftw_plan plan;
    double* in;
    fftw_complex* out;
};

/**
 * @brief A FFT struct containing the information that
 * c2r inverse FFT needs to run correctly.
 */
struct ifft_struct {
    fftw_plan plan;
    fftw_complex* in;
    double* out;
};

struct ffts {
    fftw_plan fft_plan;
    fftw_plan ifft_plan;
};

// ffts init_ffts_fftw(int b);

fft_struct init_fft_struct_fftw(int b, double* in, Complex* out);
ifft_struct init_ifft_struct_fftw(int b, Complex* in, double* out);
void fft_fftw(fft_struct info, int in_offset, int out_offset);
void ifft_fftw(ifft_struct info, int in_offset, int out_offset);
void clean_fft_fftw(fft_struct info);
void clean_ifft_fftw(ifft_struct info);

fft_plan init_fft_fftw(int b, double* in, Complex* out);
fft_plan init_ifft_fftw(int b, Complex* in, double* out);
void execute_fft_fftw(fft_plan& plan, double* in, Complex* out);
void execute_ifft_fftw(fft_plan& plan, Complex* in, double* out);
void clean_fftw(fft_plan& plan);

#endif

/**
 * @brief Initializes the use of some FFT implementation
 *
 * @param b is the length of the the output array
 * @param in is the input array location from where FFT will be computed
 * @param out is the output array location, to where the FFT computation will be stored
 * @return fft_struct A struct containing sufficient information for FFT to be computed
 */
fft_struct init_fft_struct(int b, double* in, Complex* out);

/**
 * @brief Initializes the use of some inverse FFT implementation
 *
 * @param b is the length of the the output array
 * @param in is the input array location from where inverse FFT will be computed
 * @param out is the output array location, to where the inverse FFT computation will be stored
 * @return fft_struct A struct containing sufficient information for inverse FFT to be computed
 */
ifft_struct init_ifft_struct(int b, Complex* in, double* out);

/**
 * @brief Computes the FFT.
 *
 * @param info contains the information needed for the FFT to be computed
 * @param in_offset is the offset of the input array. This is used for when you want to
 *                  compute several FFTs in parallel
 * @param out_offset is the offset of the output array. This is used for when you want to
 *                  compute several FFTs in parallel
 */
void fft(fft_struct info, int in_offset, int out_offset);

/**
 * @brief Computes the inverse FFT.
 *
 * @param info contains the information needed for the inverse FFT to be computed
 * @param in_offset is the offset of the input array. This is used for when you want to
 *                  compute several inverse FFTs in parallel
 * @param out_offset is the offset of the output array. This is used for when you want to
 *                  compute several inverse FFTs in parallel
 */
void ifft(ifft_struct info, int in_offset, int out_offset);

/**
 * @brief Frees the memory allocated to computing a FFT
 *
 * @param info contains the information needed for the FFT to be computed
 */
void clean_fft(fft_struct info);

/**
 * @brief Frees the memory allocated to computing an inverse FFT
 *
 * @param info contains the information needed for the inverse FFT to be computed
 */
void clean_ifft(ifft_struct info);

fft_plan init_fft(int b, double* in, Complex* out);
fft_plan init_ifft(int b, Complex* in, double* out);
void execute_fft(fft_plan& plan, double* in, Complex* out);
void execute_ifft(fft_plan& plan, Complex* in, double* out);
void clean(fft_plan& plan);

}  // namespace fft

#endif