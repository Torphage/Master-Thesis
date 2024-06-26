#include "fft.hpp"

namespace fft {

#ifdef USE_MKL

fft_struct init_fft_struct_mkl(int b, double* in, Complex* out) {  // GREEN GREEN WHAT IS YOUR PROBLEM GREEN
    DFTI_DESCRIPTOR_HANDLE descriptor;

    bool valid = (DFTI_NO_ERROR == DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_REAL, 1, b) &&  // Specify size and precision
                  DFTI_NO_ERROR == DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE) &&       // Out of place FFT
                  // make clear that the result should be a vector of Complex:
                  DFTI_NO_ERROR == DftiSetValue(descriptor, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX) &&
                  DFTI_NO_ERROR == DftiCommitDescriptor(descriptor));  // Finalize the descriptor

    if (!valid) {
        DftiFreeDescriptor(&descriptor);
        descriptor = nullptr;
    }

    return {
        in,
        out,
        descriptor,
    };
}

ifft_struct init_ifft_struct_mkl(int b, Complex* in, double* out) {  // GREEN GREEN WHAT IS YOUR PROBLEM GREEN
    DFTI_DESCRIPTOR_HANDLE descriptor;

    bool valid = (DFTI_NO_ERROR == DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_REAL, 1, b) &&  // Specify size and precision
                  DFTI_NO_ERROR == DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE) &&       // Out of place FFT
                  // make clear that the result should be a vector of Complex:
                  DFTI_NO_ERROR == DftiSetValue(descriptor, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX) &&
                  // chosen normalization is fft(constant)[0] = constant:
                  // DFTI_NO_ERROR == DftiSetValue(descriptor, DFTI_BACKWARD_SCALE, 1. / b) &&
                  DFTI_NO_ERROR == DftiCommitDescriptor(descriptor));  // Finalize the descriptor

    if (!valid) {
        DftiFreeDescriptor(&descriptor);
        descriptor = nullptr;
    }

    return {
        in,
        out,
        descriptor,
    };
}

void fft_mkl(fft_struct info, int in_offset, int out_offset) {
    DftiComputeForward(info.descriptor, info.in + in_offset, info.out + out_offset);
}

void ifft_mkl(ifft_struct info, int in_offset, int out_offset) {
    DftiComputeBackward(info.descriptor, info.in + in_offset, info.out + out_offset);
}

void clean_fft_mkl(fft_struct info) {
    DftiFreeDescriptor(&info.descriptor);
}

void clean_ifft_mkl(ifft_struct info) {
    DftiFreeDescriptor(&info.descriptor);
}

fft_plan init_fft_mkl(int b) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    bool valid = (DFTI_NO_ERROR == DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_REAL, 1, b) &&  // Specify size and precision
                  DFTI_NO_ERROR == DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE) &&       // Out of place FFT
                  // make clear that the result should be a vector of Complex:
                  DFTI_NO_ERROR == DftiSetValue(descriptor, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX) &&
                  // chosen normalization is fft(constant)[0] = constant:
                  // DFTI_NO_ERROR == DftiSetValue(descriptor, DFTI_BACKWARD_SCALE, 1. / b) &&
                  DFTI_NO_ERROR == DftiCommitDescriptor(descriptor));  // Finalize the descriptor

    if (!valid) {
        DftiFreeDescriptor(&descriptor);
        descriptor = nullptr;
    }

    return descriptor;
}

fft_plan init_ifft_mkl(int b) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    bool valid = (DFTI_NO_ERROR == DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_REAL, 1, b) &&  // Specify size and precision
                  DFTI_NO_ERROR == DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE) &&       // Out of place FFT
                  // make clear that the result should be a vector of Complex:
                  DFTI_NO_ERROR == DftiSetValue(descriptor, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX) &&
                  // chosen normalization is fft(constant)[0] = constant:
                  // DFTI_NO_ERROR == DftiSetValue(descriptor, DFTI_BACKWARD_SCALE, 1. / b) &&
                  DFTI_NO_ERROR == DftiCommitDescriptor(descriptor));  // Finalize the descriptor

    if (!valid) {
        DftiFreeDescriptor(&descriptor);
        descriptor = nullptr;
    }

    return descriptor;
}

void execute_fft_mkl(fft_plan& plan, double* in, Complex* out) {
    DftiComputeForward(plan, in, out);
}

void execute_ifft_mkl(fft_plan& plan, Complex* in, double* out) {
    DftiComputeBackward(plan, in, out);
}

void clean_mkl(fft_plan& plan) {
    DftiFreeDescriptor(&plan);
}

#else

fft_struct init_fft_struct_fftw(int b, double* in, Complex* out) {
    fftw_plan plan = fftw_plan_dft_r2c_1d(b, in, reinterpret_cast<fftw_complex*>(out), FFTW_DESTROY_INPUT | FFTW_PATIENT);

    return {
        plan,
        in,
        reinterpret_cast<fftw_complex*>(out),
    };
}

ifft_struct init_ifft_struct_fftw(int b, Complex* in, double* out) {
    fftw_complex* in_size = fftw_alloc_complex(b / 2 + 1);
    fftw_plan plan = fftw_plan_dft_c2r_1d(b, in_size, out, FFTW_DESTROY_INPUT | FFTW_PATIENT);

    return {
        plan,
        reinterpret_cast<fftw_complex*>(in),
        out,
    };
}

void fft_fftw(fft_struct info, int in_offset, int out_offset) {
    return fftw_execute_dft_r2c(info.plan, info.in + in_offset, info.out + out_offset);
}

void ifft_fftw(ifft_struct info, int in_offset, int out_offset) {
    return fftw_execute_dft_c2r(info.plan, info.in + in_offset, info.out + out_offset);
}

void clean_fft_fftw(fft_struct info) {
    fftw_destroy_plan(info.plan);
}

void clean_ifft_fftw(ifft_struct info) {
    fftw_destroy_plan(info.plan);
}

fft_plan init_fft_fftw(int b, double* in, Complex* out) {
    return fftw_plan_dft_r2c_1d(b, in, reinterpret_cast<fftw_complex*>(out), FFTW_DESTROY_INPUT | FFTW_PATIENT);
}

fft_plan init_ifft_fftw(int b, Complex* in, double* out) {
    return fftw_plan_dft_c2r_1d(b, reinterpret_cast<fftw_complex*>(in), out, FFTW_DESTROY_INPUT | FFTW_PATIENT);
}

void execute_fft_fftw(fft_plan& plan, double* in, Complex* out) {
    fftw_execute_dft_r2c(plan, in, reinterpret_cast<fftw_complex*>(out));
}

void execute_ifft_fftw(fft_plan& plan, Complex* in, double* out) {
    fftw_execute_dft_c2r(plan, reinterpret_cast<fftw_complex*>(in), out);
}

void clean_fftw(fft_plan& plan) {
    fftw_destroy_plan(plan);
}
#endif

fft_struct init_fft_struct(int b, double* in, Complex* out) {
#ifdef USE_MKL
    return init_fft_struct_mkl(b, in, out);
#else
    return init_fft_struct_fftw(b, in, out);
#endif
}

ifft_struct init_ifft_struct(int b, Complex* in, double* out) {
#ifdef USE_MKL
    return init_ifft_struct_mkl(b, in, out);
#else
    return init_ifft_struct_fftw(b, in, out);
#endif
}

void fft(fft_struct info, int in_offset, int out_offset) {
#ifdef USE_MKL
    return fft_mkl(info, in_offset, out_offset);
#else
    return fft_fftw(info, in_offset, out_offset);
#endif
}

void ifft(ifft_struct info, int in_offset, int out_offset) {
#ifdef USE_MKL
    return ifft_mkl(info, in_offset, out_offset);
#else
    return ifft_fftw(info, in_offset, out_offset);
#endif
}

void clean_fft(fft_struct info) {
#ifdef USE_MKL
    return clean_fft_mkl(info);
#else
    return clean_fft_fftw(info);
#endif
}

void clean_ifft(ifft_struct info) {
#ifdef USE_MKL
    return clean_ifft_mkl(info);
#else
    return clean_ifft_fftw(info);
#endif
}

fft_plan init_fft(int b, double* in, Complex* out) {
#ifdef USE_MKL
    return init_fft_mkl(b);
#else
    return init_fft_fftw(b, in, out);
#endif
}

fft_plan init_ifft(int b, Complex* in, double* out) {
#ifdef USE_MKL
    return init_ifft_mkl(b);
#else
    return init_ifft_fftw(b, in, out);
#endif
}

void execute_fft(fft_plan& plan, double* in, Complex* out) {
#ifdef USE_MKL
    return execute_fft_mkl(plan, in, out);
#else
    return execute_fft_fftw(plan, in, out);
#endif
}

void execute_ifft(fft_plan& plan, Complex* in, double* out) {
#ifdef USE_MKL
    return execute_ifft_mkl(plan, in, out);
#else
    return execute_ifft_fftw(plan, in, out);
#endif
}

void clean(fft_plan& plan) {
#ifdef USE_MKL
    return clean_mkl(plan);
#else
    return clean_fftw(plan);
#endif
}

}  // namespace fft
