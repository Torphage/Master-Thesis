
#include "../include/cxxopts.hpp"
#include "../src/compressed_mul.hpp"

#include <iostream>
#include <random>

int main(int argc, char** argv) {
    cxxopts::Options options("parameters", "Parameters to test with");

    options.allow_unrecognised_options();

    // clang-format off
    options.add_options()
        ("n,size", "Size", cxxopts::value<int>())
        ("s,seed", "The random seed", cxxopts::value<unsigned int>())
        ("p,density", "Density", cxxopts::value<double>()->default_value("1.0"))
        ("h,debug", "Enable debugging", cxxopts::value<bool>()->default_value("false"))
    ;
    // clang-format on

    options.parse_positional({"n", "density", "seed"});

    cxxopts::ParseResult result = options.parse(argc, argv);

    const int n = result["n"].as<int>();

    unsigned long seed;
    if (result.count("seed")) {
        seed = result["seed"].as<unsigned int>();
    } else {
        seed = std::random_device{}();
    }
    std::mt19937_64 rng(seed);

    double density;
    if (result.count("density")) {
        density = result["density"].as<double>();
    } else {
        density = 5 / n;  // Might need to be adjusted
    }

    MatrixRXd m1 = sparse_matrix_generator(n, density, rng);
    MatrixRXd m2 = sparse_matrix_generator(n, density, rng);

    std::cout.write(reinterpret_cast<const char*>(m1.data()), n * n * sizeof(double));
    std::cout.write(reinterpret_cast<const char*>(m2.data()), n * n * sizeof(double));
}