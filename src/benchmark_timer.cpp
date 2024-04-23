#include "benchmark_timer.hpp"
#include "../benchmark/benchmark_json.hpp"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>

namespace benchmark_timer {

void print_header() {
    std::cout << std::left << std::setw(40) << "benchmark name";
    std::cout << std::left << std::setw(15) << "samples";
    std::cout << std::left << std::setw(15) << "warmups";
    std::cout << std::left << std::setw(15) << "est run time" << std::endl;
    std::cout << std::left << std::setw(40) << "";
    std::cout << std::left << std::setw(15) << "mean";
    std::cout << std::left << std::setw(15) << "low mean";
    std::cout << std::left << std::setw(15) << "high mean" << std::endl;
    std::cout << std::left << std::setw(40) << "";
    std::cout << std::left << std::setw(15) << "median";
    std::cout << std::left << std::setw(15) << "std dev" << std::endl;
    std::cout << "--------------------------------------------------------------------------------------" << std::endl;
}

std::stringstream suitable_prefix(double num) {
    double val = 0;
    if (num > 3600.0) {
        val = num / 3600.0;
    } else if (num > 60.0) {
        val = num / 60.0;
    } else if (num >= 1.0 && num <= 60.0) {
        val = num;
    } else if (num < 1.0) {
        val = num * 1000.0;
    } else if (num < 0.001) {
        val = num * 1000000.0;
    } else if (num < 0.000001) {
        val = num * 1000000000.0;
    }

    // std::cout << val << "  " << std::fmod(val, 1000.0) << std::endl;
    int precision;
    if (std::fmod(val, 1000.0) < 10) {
        precision = 5;
    } else if (std::fmod(val, 1000.0) < 100) {
        precision = 4;
    } else {
        precision = 3;
    }

    std::stringstream ss;

    std::string postfix = "";
    if (num > 3600.0) {
        postfix = " h";
    } else if (num > 60.0) {
        postfix = " min";
    } else if (num >= 1.0 && num <= 60.0) {
        postfix = " s";
    } else if (num < 1.0) {
        postfix = " ms";
    } else if (num < 0.001) {
        postfix = " us";
    } else if (num < 0.000001) {
        postfix = " ns";
    }
    ss << std::fixed << std::setprecision(precision) << val << postfix;
    return ss;
}

void print_pre_run_info(benchmark_json::config_information& config_info, const double time) {
    if (config_info.name.size() > 37) {
        std::cout << std::left << std::setw(38) << config_info.name << std::endl
                  << std::left << std::setw(38) << "";
    } else {
        std::cout << std::left << std::setw(38) << config_info.name;
    }
    std::cout << std::right << std::setw(15) << config_info.samples;
    std::cout << std::right << std::setw(15) << config_info.warmup_iterations;
    std::cout << std::right << std::setw(15) << suitable_prefix(time).str() << std::endl;
}

void print_benchmark(benchmark_json::config_information& config_info) {
    std::cout << std::left << std::setw(38) << "   n:" + std::to_string(config_info.n) + "   b:" + std::to_string(config_info.b) + "   d:" + std::to_string(config_info.d) + "   id:" + std::to_string(config_info.matrix_id);
    std::cout << std::right << std::setw(15) << suitable_prefix(config_info.results.mean_val).str();
    std::cout << std::right << std::setw(15) << suitable_prefix(config_info.results.low_mean_val).str();
    std::cout << std::right << std::setw(15) << suitable_prefix(config_info.results.high_mean_val).str() << std::endl;
    std::cout << std::right << std::setw(38 + 15) << suitable_prefix(config_info.results.median_val).str();
    std::cout << std::right << std::setw(15) << suitable_prefix(config_info.results.std_dev_val).str() << std::endl
              << std::endl;
}

}  // namespace benchmark_timer
