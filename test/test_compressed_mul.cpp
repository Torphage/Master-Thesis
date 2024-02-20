#include <Eigen/Dense>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <iostream>
#include <random>
#include <vector>

#include "../src/compressed_mul.hpp"
#include "../src/hashing.hpp"
#include "../src/utils.hpp"
#include "../src/variance.hpp"

void test_compressed_mul(int n, int b, int d, BaseHash &hashes, BaseHash &large_hashes, std::string s) {
    Eigen::MatrixXd m1;
    Eigen::MatrixXd m2;
    Eigen::MatrixXd compressed;
    Eigen::MatrixXd result;
    Eigen::MatrixXd expected;

    SECTION(s) {
        GIVEN("Two zero-matrices") {
            m1 = Eigen::MatrixXd::Zero(n, n);
            m2 = Eigen::MatrixXd::Zero(n, n);
            expected = Eigen::MatrixXd::Zero(n, n);
            compressed = compressed_product(m1, m2, hashes);
            result = decompress_matrix(compressed, n, hashes);
            REQUIRE(result.isApprox(expected));
        }

        GIVEN("One zero-matrix") {
            m1 = Eigen::MatrixXd::Zero(n, n);
            m2 = Eigen::MatrixXd::Random(n, n);
            expected = Eigen::MatrixXd::Zero(n, n);
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
            m1 = Eigen::MatrixXd::Random(n, n);
            m2 = Eigen::MatrixXd::Random(n, n);

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
    FullyRandomHash large_random_hash(n, n*n, n*n, rng);

    test_compressed_mul(n, b, d, random_hash, large_random_hash, "Fully random hash");

    MultiplyShiftHash shift_hash(b, d, rng);
    MultiplyShiftHash large_shift_hash(n*n, n*n, rng);

    test_compressed_mul(n, b, d, shift_hash, large_shift_hash, "Multiply-shift hash");

    TabulationHash tabulation_hash(32, 32, 8, b, d, rng);
    TabulationHash large_tabulation_hash(32, 32, 8, n*n, n*n, rng);

    test_compressed_mul(n, b, d, tabulation_hash, large_tabulation_hash, "Tabulation hash");

    
}

TEST_CASE("Checking the variance bounds of the whole algorithm") {
    uint64_t N_SAMPLES = 2000000;

    int n = 8, b = 4, d = 1;

    unsigned int seed = std::random_device{}();
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> uni(-1.0, 1.0);

    Eigen::MatrixXd m1;
    Eigen::MatrixXd m2;
    HashInfo hash_info;

    GIVEN("Two uniformly distributed matrices") {
        m1 = Eigen::MatrixXd::NullaryExpr(n, n, [&]() { return uni(rng); });
        m2 = Eigen::MatrixXd::NullaryExpr(n, n, [&]() { return uni(rng); });
        SECTION("Fully-Random hash") {
            hash_info = {"FullyRandomHash", rng, b, d, n, 0, 0, 0};
            bool bound_hold = test_variance(m1, m2, hash_info, N_SAMPLES);
            REQUIRE((true == bound_hold));
        }
        SECTION("Multiply-Shift hash") {
            // hash_info = {"FullyRandomHash", rng, b, d, 0, 0, 0, 0};
            // bool bound_hold = test_variance(m1, m2, hash_info, N_SAMPLES);
            // REQUIRE((true == bound_hold));
        }
        SECTION("Tabulation hash") {
            // hash_info = {"TabulationHash", rng, b, d, 0, p, q, r};
            // bool bound_hold = test_variance(m1, m2, hash_info, N_SAMPLES);
            // REQUIRE((true == bound_hold));
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


SCENARIO("Compress") {

}

SCENARIO("Decompress") {
    
}
