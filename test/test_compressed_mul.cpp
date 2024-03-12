
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

    FullyRandomHash random_hashes(n, b, d, seed);

    compressed_mul_tests(n, b, d, "Fully random hash", random_hashes);

    MultiplyShiftHash shift_hashes(d, seed);

    compressed_mul_tests(n, b, d, "Multiply-shift hash", shift_hashes);

    int p = 32, q = 32, r = 8;
    TabulationHash tabulation_hashes(p, q, r, d, seed);

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
            auto lambda = [n, b, d](int seed) { return FullyRandomHash(n,   b,   d, seed); };
            bool bound_hold = test_variance<FullyRandomHash>(m1, m2, N_SAMPLES, MAX_SAMPLES, b, d, lambda);
            REQUIRE((true == bound_hold));
        }
        SECTION("Multiply-Shift hash") {
            std::cout << "Multiply-shift hashing variance test:" << std::endl;
            auto lambda = [d](int seed) { return MultiplyShiftHash(d, seed); };
            bool bound_hold = test_variance<MultiplyShiftHash>(m1, m2, N_SAMPLES, MAX_SAMPLES, b, d, lambda);
            REQUIRE((true == bound_hold));
        }
        SECTION("Tabulation hash") {
            int p = 32, q = 32, r = 8;
            auto lambda = [p, q, r, d](int seed) { return TabulationHash(p,   q,   r,   d, seed); };
            std::cout << "Tabulation hashing variance test:" << std::endl;
            bool bound_hold = test_variance<TabulationHash>(m1, m2, N_SAMPLES, MAX_SAMPLES, b, d, lambda);
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

    FullyRandomHash hash(n, b, d, seed);

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

    FullyRandomHash hash(n, b, d, seed);

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

    FullyRandomHash hash(n, b, d, seed);

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

    int n = 2000;
    int b = 2000;
    int d = 3;
    double density = 0.001;

    std::cout << "----- Settings -----" << std::endl;
    std::cout << "n = " << n << std::endl;
    std::cout << "b = " << b << std::endl;
    std::cout << "d = " << d << std::endl;
    std::cout << "density = " << density << std::endl;
#ifdef USE_MKL
    std::cout << "Backend = MKL" << std::endl;
#elif USE_ACCELERATE
    std::cout << "Backend = ACCELERATE" << std::endl;
#else
    std::cout << "Backend = FFTW" << std::endl;
#endif
    std::cout << "Seed = " << seed << std::endl;

    MatrixRXd compressed = MatrixRXd::Zero(d, b);
    MatrixRXd pas = MatrixRXd::Zero(d, b);
    MatrixRXd pbs = MatrixRXd::Zero(d, b);
    MatrixRXd result = MatrixRXd::Zero(n, n);
    MatrixRXd xt = MatrixRXd::Zero(n, d);

    MatrixRXd m1 = sparse_matrix_generator(n, density, rng);
    MatrixRXd m2 = sparse_matrix_generator(n, density, rng);

    FullyRandomHash hash(n, b, d, seed);
    // MultiplyShiftHash hash(d, seed);
    // int p = 32, q = 32, r = 8;
    // TabulationHash hash(p, q, r, d, seed);

    BENCHMARK("Eigen") {
        result = m1 * m2;
    };

    BENCHMARK("Parallel Compress") {
        bompressed_product_par(m1, m2, b, d, hash, compressed, pas, pbs);
    };

    BENCHMARK("Parallel Decompress") {
        debompress_matrix_par(compressed, n, b, d, hash, result, xt);
    };

    // BENCHMARK("Sequential Compress") {
    //     compressed1 = compressed_product(m1, m2, b, d, hash);
    // };
    // BENCHMARK("Sequential Decompress") {
    //     result1 = decompress_matrix(compressed1, n, b, d, hash);
    // };
}