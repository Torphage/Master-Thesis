// Your First C++ Program
#include <catch2/catch_test_macros.hpp>

#include <iostream>

TEST_CASE("Section showcase") {
    std::cout << '1';
    SECTION("A") {
        std::cout << 'A';
        SECTION("a") { std::cout << 'a'; }
        SECTION("b") { std::cout << 'b'; }
    }
    SECTION("B") {
        std::cout << 'B';
        SECTION("a") { std::cout << 'a'; }
        SECTION("b") { std::cout << 'b'; }
    }
    std::cout << '\n';
}