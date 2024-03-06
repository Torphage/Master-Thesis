
#include "test_compressed_mul.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "../src/compressed_mul.hpp"
#include "../src/utils.hpp"
#include "../src/variance.hpp"

TEST_CASE("Compressed multiplication tests") {
    unsigned int seed = std::random_device{}();
    std::mt19937_64 rng(seed);

    int n = 16;
    int b = 4;
    int d = 2;

    Hashes<Eigen::MatrixXi> random_hashes = fully_random_constructor(n, b, d, rng);

    compressed_mul_tests(n, b, d, "Fully random hash", fully_random_hash(), random_hashes);

    Hashes<MatrixXui> shift_hashes = multiply_shift_constructor(d, rng);

    compressed_mul_tests(n, b, d, "Multiply-shift hash", multiply_shift_hash(), shift_hashes, b);

    Hashes<std::vector<MatrixXui>> tabulation_hashes = tabulation_constructor(32, 32, 8, d, rng);

    compressed_mul_tests(n, b, d, "Tabulation hash", tabulation_hash(), tabulation_hashes, b, 8, 4);
}

TEST_CASE("Checking the variance bounds of the whole algorithm") {
    int N_SAMPLES = 1000;
    int MAX_SAMPLES = 100000;

    int n = 5, b = 200, d = 1;

    unsigned int seed = std::random_device{}();
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> uni(0, 1.0);

    MatrixRXd m1;
    MatrixRXd m2;

    GIVEN("Two uniformly distributed matrices") {
        m1 = MatrixRXd::NullaryExpr(n, n, [&]() { return uni(rng); });
        m2 = MatrixRXd::NullaryExpr(n, n, [&]() { return uni(rng); });
        SECTION("Fully-Random hash") {
            std::tuple<int, int, int, std::mt19937_64> cargs = std::make_tuple(n, b, d, rng);
            std::cout << "Fully random hashing variance test:" << std::endl;
            auto start = std::chrono::steady_clock::now();
            bool bound_hold = test_variance2<Eigen::MatrixXi>(m1, m2, N_SAMPLES, MAX_SAMPLES, b, d, fully_random_constructor, fully_random_hash(), cargs);
            std::cout << "\nElapsed(ms)=" << since(start).count();
            REQUIRE((true == bound_hold));
            std::cout << std::endl;
        }
        SECTION("Multiply-Shift hash") {
            std::tuple<int, std::mt19937_64> cargs = std::make_tuple(d, rng);
            std::cout << "Multiply-shift hashing variance test:" << std::endl;
            auto start = std::chrono::steady_clock::now();
            bool bound_hold = test_variance2<MatrixXui>(m1, m2, N_SAMPLES, MAX_SAMPLES, b, d, multiply_shift_constructor, multiply_shift_hash(), cargs, b);
            std::cout << "\nElapsed(ms)=" << since(start).count();
            REQUIRE((true == bound_hold));
            std::cout << std::endl;
        }
        SECTION("Tabulation hash") {
            int p = 32, q = 32, r = 8;
            std::tuple<int, int, int, int, std::mt19937_64> cargs = std::make_tuple(p, q, r, d, rng);
            std::cout << "Tabulation hashing variance test:" << std::endl;
            auto start = std::chrono::steady_clock::now();
            bool bound_hold = test_variance2<std::vector<MatrixXui>>(m1, m2, N_SAMPLES, MAX_SAMPLES, b, d, tabulation_constructor, tabulation_hash(), cargs, b, r, ceil(p / r));
            std::cout << "\nElapsed(ms)=" << since(start).count();
            REQUIRE((true == bound_hold));
            std::cout << std::endl;
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
    std::mt19937_64 rng(seed);

    int n = 60;
    int b = 20;
    int d = 8;
    MatrixRXd m1;
    MatrixRXd m2;
    MatrixRXd compressed;
    MatrixRXd result;
    MatrixRXd expected;

    Hashes<Eigen::MatrixXi> hashes = fully_random_constructor(n, b, d, rng);

    SECTION("Compressing two zero matrices") {
        m1 = MatrixRXd::Zero(n, n);
        m2 = MatrixRXd::Zero(n, n);
        expected = MatrixRXd::Zero(d, b);

        result = compressed_product(m1, m2, b, d, fully_random_hash(), hashes);

        REQUIRE(result.isApprox(expected));
    }

    SECTION("Compressing one zero matrix") {
        m1 = MatrixRXd::Zero(n, n);
        m2 = MatrixRXd::Random(n, n);
        expected = MatrixRXd::Zero(d, b);

        result = compressed_product(m1, m2, b, d, fully_random_hash(), hashes);

        REQUIRE(result.isApprox(expected));
    }
}

TEST_CASE("Decompress") {
    unsigned int seed = std::random_device{}();
    std::mt19937_64 rng(seed);

    int n = 60;
    int b = 20;
    int d = 8;
    MatrixRXd m1;
    MatrixRXd m2;
    MatrixRXd compressed;
    MatrixRXd result;
    MatrixRXd expected;

    Hashes<Eigen::MatrixXi> hashes = fully_random_constructor(n, b, d, rng);

    GIVEN("A matrix with no zero-elements") {
        compressed = MatrixRXd::Random(n, n);  // make sure that no zero elements are present
        THEN("No zero-elements are present in the output") {
            result = decompress_matrix(compressed, n, b, d, fully_random_hash(), hashes);
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
    std::mt19937_64 rng(seed);

    Hashes<Eigen::MatrixXi> hashes = fully_random_constructor(n, b, d, rng);

    SECTION("Parallel compress gives same as sequential compress") {
        m1 = MatrixRXd::Random(n, n);
        m2 = MatrixRXd::Random(n, n);

        expected = compressed_product(m1, m2, b, d, fully_random_hash(), hashes);
        result = compressed_product_par(m1, m2, b, d, fully_random_hash(), hashes);

        REQUIRE(result.isApprox(expected));
    }

    SECTION("Parallel decompress gives same as sequential decompress") {
        m1 = MatrixRXd::Random(n, n);
        m2 = MatrixRXd::Random(n, n);

        compressed = compressed_product(m1, m2, b, d, fully_random_hash(), hashes);
        expected = decompress_matrix(compressed, n, b, d, fully_random_hash(), hashes);
        result = decompress_matrix_par(compressed, n, b, d, fully_random_hash(), hashes);

        REQUIRE(result.isApprox(expected));
    }
}

TEST_CASE("Benchmarks", "[!benchmark]") {
    unsigned int seed = std::random_device{}();
    std::mt19937_64 rng(seed);

    int n = 700;
    int b = 600;
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

    MatrixRXd compressed1;
    MatrixRXd compressed2;
    MatrixRXd result1;
    MatrixRXd result2;

    MatrixRXd m1 = sparse_matrix_generator(n, density, rng);
    MatrixRXd m2 = sparse_matrix_generator(n, density, rng);


    Hashes<Eigen::MatrixXi> hashes = fully_random_constructor(n, b, d, rng);
    // Hashes<MatrixXui> hashes = multiply_shift_constructor(d, rng);
    // int p = 32, q = 32, r = 8, t = ceil(p/r);
    // Hashes<std::vector<MatrixXui>> hashes = tabulation_constructor(p, q, r, d, rng);

    BENCHMARK("Eigen") {
        result1 = m1 * m2;
    };

    BENCHMARK("Parallel Compress") {
        compressed2 = compressed_product_par(m1, m2, b, d, fully_random_hash(), hashes);
    };

    BENCHMARK("Parallel Decompress") {
        result2 = decompress_matrix_par(compressed2, n, b, d, fully_random_hash(), hashes);
    };

    // BENCHMARK("Sequential Compress") {
    //     compressed1 = compressed_product(m1, m2, b, d, fully_random_hash(), hashes1);
    // };
    // BENCHMARK("Sequential Decompress") {
    //     result1 = decompress_matrix(compressed1, n, b, d, fully_random_hash(), hashes1);
    // };
}