#include <cmath>
#include <catch2/catch_test_macros.hpp>
#include <iostream>


template <typename T> std::string type_name();

SCENARIO("HASHING") {
    SECTION("Fully Random") {
        // Possibly: Make sure that index out of bounds
        // returns an error, instead of choosing a random
        // value from the memory. Thinking specifically
        // of right shift as well as overflows.
    }

    SECTION("Multiply-Shift") {
        GIVEN("an uint32") {
            uint32_t x = UINT32_MAX - 5;
            WHEN("right shift") {
                THEN("gives type ???") {
                    // auto prod = (UINT32_MAX - 5) * (UINT64_MAX - 3);
                    // std::cout << prod << std::endl;
                    // std::cout << type_name<decltype((UINT16_MAX - 5) + (UINT16_MAX - 3))>() << std::endl;
                    // LLONG_MAX
                }
            }
        }
        // GIVEN("two uint64") {
        //     WHEN("right shift") {
        //         THEN("returns uint64") {
        //             uint64_t num1 = static_cast<uint64_t>(pow(2, 64));
        //         }
        //     }
    }
        // Confirm that the algorithm works, with the combination of
        // uint32 and uint64. Try with larger values, since it's only
        // then that overflows will occur.
    SECTION("Tabulation") {
        // Study its weird behavior for larger input matrices.
    }
}

