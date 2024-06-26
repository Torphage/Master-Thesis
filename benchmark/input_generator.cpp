
// #include "../include/cxxopts.hpp"
#include "../include/rapidcsv.h"
#include "../src/utils.hpp"

#include <iostream>
#include <random>

int main() {
    rapidcsv::Document doc("input.csv");

    std::vector<int> runs = doc.GetColumn<int>("run");
    std::vector<int> ns = doc.GetColumn<int>("n");
    std::vector<double> densities = doc.GetColumn<double>("density");
    std::vector<int> matrix_ids = doc.GetColumn<int>("matrix_id");
    std::vector<unsigned int> matrix_seeds = doc.GetColumn<unsigned int>("matrix_seed");

    int number_of_lines = ns.size();

    std::vector<int> ids;
    for (int index = 0; index < number_of_lines; index++) {
        int run = runs[index];
        int n = ns[index];
        double density = densities[index];
        int matrix_id = matrix_ids[index];
        unsigned int matrix_seed = matrix_seeds[index];

        if (matrix_seed == 0) matrix_seed = std::random_device{}();

        if (run == 0) continue;

        if (std::find(ids.begin(), ids.end(), matrix_id) != ids.end()) {
            continue;
        }

        ids.push_back(matrix_id);
        std::mt19937_64 rng(matrix_seed);

        MatrixRXd m1 = sparse_matrix_generator(n, density, rng);
        MatrixRXd m2 = sparse_matrix_generator(n, density, rng);

        std::cout.write(reinterpret_cast<const char*>(m1.data()), n * n * sizeof(double));
        std::cout.write(reinterpret_cast<const char*>(m2.data()), n * n * sizeof(double));
    }
}