#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "../src/utils.hpp"
#include "../src/hashing.hpp"


template <typename T, typename H, typename... Args>
void compressed_mul_tests(int n, int b, int d, std::string s, T hash, Hashes<H>& hashes, Args... args) {
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
            compressed = compressed_product(m1, m2, b, d, hash, hashes, args...);
            result = decompress_matrix(compressed, n, b, d, hash, hashes, args...);
            REQUIRE(result.isApprox(expected));
        }

        GIVEN("One zero-matrix") {
            m1 = MatrixRXd::Zero(n, n);
            m2 = MatrixRXd::Random(n, n);
            expected = MatrixRXd::Zero(n, n);
            compressed = compressed_product(m1, m2, b, d, hash, hashes, args...);
            result = decompress_matrix(compressed, n, b, d, hash, hashes, args...);
            WHEN("zero times random") {
                REQUIRE(result.isApprox(expected));
            }

            compressed = compressed_product(m1, m2, b, d, hash, hashes, args...);
            result = decompress_matrix(compressed, n, b, d, hash, hashes, args...);
            WHEN("random times zero") {
                REQUIRE(result.isApprox(expected));
            }
        }

    }
}