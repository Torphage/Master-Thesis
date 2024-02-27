#include <Eigen/Dense>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "../src/compressed_mul.hpp"
#include "../src/hashing.hpp"
#include "../src/utils.hpp"
#include "../src/variance.hpp"

void test_compressed_mul(int n, int b, int d, BaseHash &hashes, BaseHash &large_hashes, std::string s) {
    MatrixRXd m1;
    MatrixRXd m2;
    MatrixRXd compressed;
    MatrixRXd result;
    MatrixRXd expected;

    SECTION(s) {
        GIVEN("Two zero-matrices") {
            m1 = MatrixRXd::Zero(n, n);
            m2 = MatrixRXd::Zero(n, n);
            expected = MatrixRXd::Zero(n, n);
            compressed = compressed_product(m1, m2, hashes);
            result = decompress_matrix(compressed, n, hashes);
            REQUIRE(result.isApprox(expected));
        }

        GIVEN("One zero-matrix") {
            m1 = MatrixRXd::Zero(n, n);
            m2 = MatrixRXd::Random(n, n);
            expected = MatrixRXd::Zero(n, n);
            compressed = compressed_product(m1, m2, hashes);
            result = decompress_matrix(compressed, n, hashes);
            WHEN("zero times random") {
                REQUIRE(result.isApprox(expected));
            }

            compressed = compressed_product(m2, m1, hashes);
            result = decompress_matrix(compressed, n, hashes);
            WHEN("random times zero") {
                REQUIRE(result.isApprox(expected));
            }
        }

        GIVEN("Two non-zero matrices") {
            m1 = MatrixRXd::Random(n, n);
            m2 = MatrixRXd::Random(n, n);

            expected = m1 * m2;

            compressed = compressed_product(m1, m2, large_hashes);
            result = decompress_matrix(compressed, n, large_hashes);
            WHEN("random times random, correct with prob (very very very close to) 1, (should almost never fail)") {
                REQUIRE(result.isApprox(expected));
            }
        }
    }
}

TEST_CASE("Compressed multiplication tests") {
    unsigned int seed = std::random_device{}();
    std::mt19937_64 rng(seed);

    int n = 16;
    int b = 4;
    int d = 2;

    FullyRandomHash random_hash(n, b, d, rng);
    FullyRandomHash large_random_hash(n, n * n, n * n, rng);

    test_compressed_mul(n, b, d, random_hash, large_random_hash, "Fully random hash");

    MultiplyShiftHash shift_hash(b, d, rng);
    MultiplyShiftHash large_shift_hash(n * n, n * n, rng);

    test_compressed_mul(n, b, d, shift_hash, large_shift_hash, "Multiply-shift hash");

    TabulationHash tabulation_hash(32, 32, 8, b, d, rng);
    TabulationHash large_tabulation_hash(32, 32, 8, n * n, n * n, rng);

    test_compressed_mul(n, b, d, tabulation_hash, large_tabulation_hash, "Tabulation hash");
}

TEST_CASE("Checking the variance bounds of the whole algorithm") {
    uint64_t N_SAMPLES = 400000;

    int n = 6, b = 4, d = 1;

    unsigned int seed = std::random_device{}();
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> uni(0, 1.0);

    MatrixRXd m1;
    MatrixRXd m2;
    HashInfo hash_info;

    GIVEN("Two uniformly distributed matrices") {
        m1 = MatrixRXd::NullaryExpr(n, n, [&]() { return uni(rng); });
        m2 = MatrixRXd::NullaryExpr(n, n, [&]() { return uni(rng); });
        SECTION("Fully-Random hash") {
            hash_info = {"FullyRandomHash", rng, b, d, n, 0, 0, 0};
            bool bound_hold = test_variance(m1, m2, hash_info, N_SAMPLES);
            REQUIRE((true == bound_hold));
            std::cout << std::endl;
        }
        SECTION("Multiply-Shift hash") {
            hash_info = {"MultiplyShiftHash", rng, b, d, 0, 0, 0, 0};
            bool bound_hold = test_variance(m1, m2, hash_info, N_SAMPLES);
            REQUIRE((true == bound_hold));
            std::cout << std::endl;
        }
        SECTION("Tabulation hash") {
            int p = 32, q = 32, r = 8;
            hash_info = {"TabulationHash", rng, b, d, 0, p, q, r};
            bool bound_hold = test_variance(m1, m2, hash_info, N_SAMPLES);
            REQUIRE((true == bound_hold));
            std::cout << std::endl;
        }
    }
    GIVEN("Two sparse matrices") {
        m1 = sparse_matrix_generator(n, 0.25, rng);  // TODO: make sure sparse_matrix_generator() works
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
}

TEST_CASE("Decompress") {
}

TEST_CASE("Parallel") {
    int n = 100;
    int b = 20;
    int d = 28;

    MatrixRXd m1;
    MatrixRXd m2;
    MatrixRXd compressed;
    MatrixRXd result;
    MatrixRXd expected;

    unsigned int seed = std::random_device{}();
    std::mt19937_64 rng(seed);

    FullyRandomHash hashes(n, b, d, rng);

    SECTION("Parallel compress gives same as sequential compress") {
        m1 = MatrixRXd::Random(n, n);
        m2 = MatrixRXd::Random(n, n);

        expected = compressed_product(m1, m2, hashes);
        result = compressed_product_par(m1, m2, hashes);

        REQUIRE(result.isApprox(expected));
    }

    SECTION("Parallel decompress gives same as sequential decompress") {
        m1 = MatrixRXd::Random(n, n);
        m2 = MatrixRXd::Random(n, n);

        compressed = compressed_product(m1, m2, hashes);
        expected = decompress_matrix(compressed, n, hashes);
        result = decompress_matrix_par(compressed, n, hashes);

        REQUIRE(result.isApprox(expected));
    }
}

TEST_CASE("Benchmarks", "[!benchmark]") {
    int n = 100;
    int b = 10;
    int d = 8;

    unsigned int seed = std::random_device{}();
    std::mt19937_64 rng(seed);

    MatrixRXd compressed1;
    MatrixRXd compressed2;
    MatrixRXd result1;
    MatrixRXd result2;

    MatrixRXd m1 = sparse_matrix_generator(n, 0.001, rng);
    MatrixRXd m2 = sparse_matrix_generator(n, 0.001, rng);

    FullyRandomHash hashes(n, b, d, rng);
    // MultiplyShiftHash hashes(b,d,rng);

    BENCHMARK("Eigen") {
        result1 = m1 * m2;
    };

    BENCHMARK("Parallel Compress") {
        compressed2 = compressed_product_par(m1, m2, hashes);
    };

    BENCHMARK("Parallel Decompress") {
        result2 = decompress_matrix_par(compressed2, n, hashes);
    };

    // BENCHMARK("Sequential Compress") {
    //     compressed1 = compressed_product(m1, m2, hashes);
    // };
    // BENCHMARK("Sequential Decompress") {
    //     result1 = decompress_matrix(compressed1, n, hashes);
    // };

    BENCHMARK("FullyRandomHash constructor") {
        FullyRandomHash hashes(n, b, d, rng);
    };

    BENCHMARK("MultiplyShiftHash constructor") {
        MultiplyShiftHash hashes(b, d, rng);
    };

    BENCHMARK("TabulationHash constructor") {
        int p = 32, q = 32, r = 8;
        TabulationHash hashes(p, q, r, b, d, rng);
    };

    BENCHMARK("find_median") {
        Eigen::VectorXd vec = Eigen::VectorXd::Random(d);
        double val;
        #pragma omp parallel for
        for (int i = 0; i < n*n; i++) {
            val = find_median(vec);
        }
    };

    // This is half of our bloody runtime innit
    BENCHMARK("Access one hash function, the number of times it is accessed in compressed_product") {
        int temp;
#pragma omp parallel for
        for (int t = 0; t < d; t++) {
            for (int k = 0; k < n; k++) {
                for (int i = 0; i < n; i++) {
                    temp = hashes.hash(0, t, i);
                }
            }
        }
    };

    SECTION("") {
    }
}