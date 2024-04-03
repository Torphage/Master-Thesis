
#include "test_compressed_mul.hpp"

#include "../src/compressed_mul.hpp"
#include "../src/utils.hpp"
#include "../src/variance.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <iostream>
#include <memory>
#include <random>

TEST_CASE("Compressed multiplication tests") {
    unsigned int seed = std::random_device{}();
    int n = 16;
    int b = 4;
    int d = 2;

    FullyRandomHash<int> random_hashes(n, b, d, seed);

    compressed_mul_tests(n, b, d, "Fully random hash", random_hashes);

    MultiplyShiftHash<uint64_t, uint32_t> shift_hashes(d, seed);

    compressed_mul_tests(n, b, d, "Multiply-shift hash", shift_hashes);

    TabulationHash<uint32_t, uint32_t, 8> tabulation_hashes(d, seed);

    compressed_mul_tests(n, b, d, "Tabulation hash", tabulation_hashes);
}

TEST_CASE("Checking the variance bounds of the whole algorithm") {
    int N_SAMPLES = 1000;
    int MAX_SAMPLES = 2000000;

    int n = 5, b = 2, d = 1;

    unsigned int seed = std::random_device{}();
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> uni(0, 1.0);

    MatrixRXd m1;
    MatrixRXd m2;

    GIVEN("Two uniformly distributed matrices") {
        m1 = MatrixRXd::NullaryExpr(n, n, [&]() { return uni(rng); });
        m2 = MatrixRXd::NullaryExpr(n, n, [&]() { return uni(rng); });
        SECTION("Fully-Random hash") {
            std::cout << "Fully random hashing variance test:" << std::endl;
            auto lambda = [n, b, d](int seed) { return FullyRandomHash<int>(n, b, d, seed); };
            bool bound_hold = test_variance<FullyRandomHash<int>>(m1, m2, N_SAMPLES, MAX_SAMPLES, b, d, lambda);
            REQUIRE((true == bound_hold));
        }
        SECTION("Multiply-Shift hash") {
            std::cout << "Multiply-shift hashing variance test:" << std::endl;
            auto lambda = [d](int seed) { return MultiplyShiftHash<uint64_t, uint32_t>(d, seed); };
            bool bound_hold = test_variance<MultiplyShiftHash<uint64_t, uint32_t>>(m1, m2, N_SAMPLES, MAX_SAMPLES, b, d, lambda);
            REQUIRE((true == bound_hold));
        }
        SECTION("Tabulation hash") {
            auto lambda = [d](int seed) { return TabulationHash<uint32_t, uint32_t, 8>(d, seed); };
            std::cout << "Tabulation hashing variance test:" << std::endl;
            bool bound_hold = test_variance<TabulationHash<uint32_t, uint32_t, 8>>(m1, m2, N_SAMPLES, MAX_SAMPLES, b, d, lambda);
            REQUIRE((true == bound_hold));
        }
    }
    GIVEN("Two sparse matrices") {
        m1 = sparse_matrix_generator(n, 0.25, rng);
        m2 = sparse_matrix_generator(n, 0.25, rng);
        SECTION("Fully-Random hash") {
        }
        SECTION("Multiply-Shift hash") {
        }
        SECTION("Tabulation hash") {
        }
    }
}

TEST_CASE("Compress") {
    unsigned int seed = std::random_device{}();

    int n = 60;
    int b = 20;
    int d = 8;
    MatrixRXd m1;
    MatrixRXd m2;
    MatrixRXd compressed;
    MatrixRXd result;
    MatrixRXd expected;

    FullyRandomHash<int> hash(n, b, d, seed);

    SECTION("Compressing two zero matrices") {
        m1 = MatrixRXd::Zero(n, n);
        m2 = MatrixRXd::Zero(n, n);
        expected = MatrixRXd::Zero(d, b);

        result = compressed_product(m1, m2, b, d, hash);

        REQUIRE(result.isApprox(expected));
    }

    SECTION("Compressing one zero matrix") {
        m1 = MatrixRXd::Zero(n, n);
        m2 = MatrixRXd::Random(n, n);
        expected = MatrixRXd::Zero(d, b);

        result = compressed_product(m1, m2, b, d, hash);

        REQUIRE(result.isApprox(expected));
    }
}

TEST_CASE("Decompress") {
    unsigned int seed = std::random_device{}();

    int n = 60;
    int b = 20;
    int d = 8;
    MatrixRXd m1;
    MatrixRXd m2;
    MatrixRXd compressed;
    MatrixRXd result;
    MatrixRXd expected;

    FullyRandomHash<int> hash(n, b, d, seed);

    GIVEN("A matrix with no zero-elements") {
        compressed = MatrixRXd::Random(n, n);  // make sure that no zero elements are present
        THEN("No zero-elements are present in the output") {
            result = decompress_matrix(compressed, n, b, d, hash);
            REQUIRE(!(result.array() == 0.0).any());
        }
    }
}

TEST_CASE("Parallel") {
    int n = 60;
    int b = 20;
    int d = 8;

    MatrixRXd m1;
    MatrixRXd m2;
    MatrixRXd compressed;
    MatrixRXd result;
    MatrixRXd expected;

    unsigned int seed = std::random_device{}();

    FullyRandomHash<int> hash(n, b, d, seed);

    SECTION("Parallel compress gives same as sequential compress") {
        m1 = MatrixRXd::Random(n, n);
        m2 = MatrixRXd::Random(n, n);

        expected = compressed_product(m1, m2, b, d, hash);
        result = compressed_product_par(m1, m2, b, d, hash);

        REQUIRE(result.isApprox(expected));
    }

    SECTION("Parallel decompress gives same as sequential decompress") {
        m1 = MatrixRXd::Random(n, n);
        m2 = MatrixRXd::Random(n, n);

        compressed = compressed_product(m1, m2, b, d, hash);
        expected = decompress_matrix(compressed, n, b, d, hash);
        result = decompress_matrix_par(compressed, n, b, d, hash);

        REQUIRE(result.isApprox(expected));
    }
}

TEST_CASE("Benchmarks", "[!benchmark]") {
    unsigned int seed = std::random_device{}();
    std::mt19937_64 rng(seed);

    int n = 1000;
    int b = 1000;
    int d = 5;
    double density = 0.001;

    {
        std::cout << "----- Settings -----" << std::endl;
        std::cout << "n = " << n << std::endl;
        std::cout << "b = " << b << std::endl;
        std::cout << "d = " << d << std::endl;
        std::cout << "density = " << density << std::endl;
#ifdef USE_MKL
        std::cout << "Backend = MKL" << std::endl;
#elif USE_ACCELERATE
        std::cout << "Backend = ACCELERATE" << std::endl;
#elif USE_AOCL
        std::cout << "Backend = AOCL" << std::endl;
#else
        std::cout << "Backend = FFTW" << std::endl;
#endif
        std::cout << "Seed = " << seed << std::endl;
    }

    MatrixRXd m1 = sparse_matrix_generator(n, density, rng);
    MatrixRXd m2 = sparse_matrix_generator(n, density, rng);

    FullyRandomHash<uint64_t> hash(n, b, d, seed);
    // MultiplyShiftHash<uint32_t, uint16_t> hash(d, seed);
    // TabulationHash<uint32_t, uint32_t, 8> hash(d, seed);

    {
        // MatrixRXd result = MatrixRXd::Zero(n, n);

        // BENCHMARK("Eigen") {
        //     result = m1 * m2;
        // };
    }

    {
        MatrixRXd compressed0 = MatrixRXd::Zero(d, b);
        MatrixRXd pas0 = MatrixRXd::Zero(d, b);
        MatrixRXd pbs0 = MatrixRXd::Zero(d, b);
        MatrixRXcd ps0 = MatrixRXcd::Zero(d, b);
        MatrixRXcd out10(d, b / 2 + 1);
        MatrixRXcd out20(d, b / 2 + 1);
        fft_struct fft10 = init_fft(b, pas0.data(), out10.data());
        fft_struct fft20 = init_fft(b, pbs0.data(), out20.data());
        ifft_struct ifft0 = init_ifft(b, ps0.data(), compressed0.data());

        BENCHMARK("Parallel Compress - Original") {
            bompressed_product_par(m1, m2, b, d, hash, compressed0, pas0, pbs0, ps0, out10, out20, fft10, fft20, ifft0);
        };

        clean_fft(fft10);
        clean_fft(fft20);
        clean_ifft(ifft0);
    }

    {
        int num_threads = std::max(d, omp_get_max_threads());
        MatrixRXd compressed = MatrixRXd::Zero(d, b);
        MatrixRXd pas = MatrixRXd::Zero(num_threads, b);
        MatrixRXd pbs = MatrixRXd::Zero(num_threads, b);
        MatrixRXcd ps = MatrixRXcd::Zero(d, b);
        MatrixRXcd out1(num_threads, b / 2 + 1);
        MatrixRXcd out2(num_threads, b / 2 + 1);
        fft_struct fft1 = init_fft(b, pas.data(), out1.data());
        fft_struct fft2 = init_fft(b, pbs.data(), out2.data());
        ifft_struct ifft1 = init_ifft(b, ps.data(), compressed.data());

        BENCHMARK("Parallel Compress - Threads") {
            bompressed_product_par_threaded(m1, m2, b, d, hash, compressed, pas, pbs, ps, out1, out2, fft1, fft2, ifft1);
        };

        clean_fft(fft1);
        clean_fft(fft2);
        clean_ifft(ifft1);
    }

    {
        int num_threads = std::max(d, omp_get_max_threads());
        MatrixRXd compressed3 = MatrixRXd::Zero(d, b);
        MatrixRXd pas3 = MatrixRXd::Zero(num_threads, b);
        MatrixRXd pbs3 = MatrixRXd::Zero(num_threads, b);
        MatrixRXcd ps3 = MatrixRXcd::Zero(d, b);
        MatrixRXcd out13(d * n, b / 2 + 1);
        MatrixRXcd out23(d * n, b / 2 + 1);
        fft_struct fft13 = init_fft(b, pas3.data(), out13.data());
        fft_struct fft23 = init_fft(b, pbs3.data(), out23.data());
        ifft_struct ifft3 = init_ifft(b, ps3.data(), compressed3.data());

        BENCHMARK("Parallel Compress - Matti (combi with threads)") {
            bompressed_product_par_large_threaded(m1, m2, b, d, hash, compressed3, pas3, pbs3, ps3, out13, out23, fft13, fft23, ifft3);
        };

        clean_fft(fft13);
        clean_fft(fft23);
        clean_ifft(ifft3);
    }

    {
        MatrixRXd compressed4 = MatrixRXd::Zero(d, b);
        MatrixRXd pas4 = MatrixRXd::Zero(d * n, b);
        MatrixRXd pbs4 = MatrixRXd::Zero(d * n, b);
        MatrixRXcd ps4 = MatrixRXcd::Zero(d, b);
        MatrixRXcd out14(d * n, b / 2 + 1);
        MatrixRXcd out24(d * n, b / 2 + 1);
        fft_struct fft14 = init_fft(b, pas4.data(), out14.data());
        fft_struct fft24 = init_fft(b, pbs4.data(), out24.data());
        ifft_struct ifft4 = init_ifft(b, ps4.data(), compressed4.data());

        BENCHMARK("Parallel Compress - Matti (collapse)") {
            bompressed_product_par_large(m1, m2, b, d, hash, compressed4, pas4, pbs4, ps4, out14, out24, fft14, fft24, ifft4);
        };

        clean_fft(fft14);
        clean_fft(fft24);
        clean_ifft(ifft4);
    }

    // BENCHMARK("Parallel Compress") {
    //     bompressed_product_par(m1, m2, b, d, hash, compressed, pas, pbs, ps, out1, out2, fft1, fft2, ifft);
    // };

    // BENCHMARK("Parallel Decompress") {
    //     debompress_matrix_par(compressed, n, b, d, hash, result, xt);
    // };

    // BENCHMARK("Sequential Compress") {
    //     compressed1 = compressed_product(m1, m2, b, d, hash);
    // };
    // BENCHMARK("Sequential Decompress") {
    //     result1 = decompress_matrix(compressed1, n, b, d, hash);
    // };
}