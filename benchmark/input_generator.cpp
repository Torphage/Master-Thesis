
#include <iostream>
#include <random>

#include "../include/cxxopts.hpp"
#include "../src/compressed_mul.hpp"

int main(int argc, char** argv) {
    cxxopts::Options options("parameters", "Parameters to test with");

    unsigned int temp_seed = std::random_device{}();
    // clang-format off
    options.add_options()
        ("s,seed", "The random seed", cxxopts::value<unsigned int>()->default_value(std::to_string(temp_seed)))
        ("n,size", "Size", cxxopts::value<int>()->default_value("0"))
        ("p,density", "Density", cxxopts::value<double>()->default_value("1.0"))
        ("h,debug", "Enable debugging", cxxopts::value<bool>()->default_value("false"))
    ;
    // clang-format on

    cxxopts::ParseResult result = options.parse(argc, argv);

    unsigned int seed = result["seed"].as<unsigned int>();
    std::mt19937_64 rng(seed);
    const int n = result["n"].as<int>();
    double density = result["density"].as<double>();

    MatrixRXd m1 = sparse_matrix_generator(n, density, rng);
    MatrixRXd m2 = sparse_matrix_generator(n, density, rng);

    std::cout.write(reinterpret_cast<const char*>(m1.data()), n * n * sizeof(double));
    std::cout.write(reinterpret_cast<const char*>(m2.data()), n * n * sizeof(double));
}