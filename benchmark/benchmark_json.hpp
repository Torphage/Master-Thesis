#ifndef BENCHMARK_JSON_HPP_
#define BENCHMARK_JSON_HPP_

#include "../include/json.hpp"

#include <string>
#include <vector>

namespace benchmark_json {

struct benchmarkinfo {
    std::vector<double> warmup_vals;
    std::vector<double> vals;
    double runtime_val;
    double mean_val;
    double low_mean_val;
    double high_mean_val;
    double median_val;
    double variance_val;
    double std_dev_val;

    benchmarkinfo() : runtime_val(0.0), mean_val(0.0), low_mean_val(0.0), high_mean_val(0.0), median_val(0.0), variance_val(0.0), std_dev_val(0.0) {
        warmup_vals = std::vector<double>(0);
        vals = std::vector<double>(0);
    }
};

struct hardware_information {
};

/**
 * @brief information about what benchmarks were run
 */
struct config_information {
    int n;
    int b;
    int d;
    double density;
    std::string name;
    std::string hash;
    std::string function;
    unsigned int matrix_id;
    unsigned int matrix_seed;
    unsigned int hash_seed;
    int samples;
    int warmup_iterations;
    int cores;
    benchmarkinfo results;

    config_information() : n(0), b(0), d(0), density(0.0), name(""), hash(""), function(""), matrix_id(0), matrix_seed(0), hash_seed(0), samples(0), warmup_iterations(0), cores(0) {
        results = benchmarkinfo();
    }
};

/**
 * @brief
 */
struct information {
    hardware_information hardware;
    std::vector<config_information> config;

    information() {
        hardware = hardware_information();
        config = std::vector<config_information>(0);
    }
};

nlohmann::json read_file(std::string filename);
void write_file(std::string filename, information info);

}  // namespace benchmark_json

#endif