#ifndef BENCHMARK_JSON_HPP_
#define BENCHMARK_JSON_HPP_

#include "../include/json.hpp"

#include <string>
#include <vector>

namespace benchmark_json {

struct benchmarkinfo {
    std::vector<double> vals;
    double meanval;
    double lowmeanval;
    double highmeanval;
    double medianval;
    double varianceval;
    double stddevval;
    double lowstddevval;
    double highstddevval;
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
    unsigned int seed;
    std::string hash;
    std::string function;
    unsigned int matrix_seed;
    unsigned int hash_seed;
    int samples;
    int warmup_iterations;
};

/**
 * @brief 
 */
struct information {
    hardware_information hardware;
    std::vector<config_information> config;
    benchmarkinfo results;
};

static void read_file(std::string filename);
static void write_file(std::string filename, information info);

}  // namespace benchmark_json

#endif