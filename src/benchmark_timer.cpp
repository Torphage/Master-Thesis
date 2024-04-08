#include "benchmark_timer.hpp"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>

namespace benchmark_timer {

void print_header() {
    std::cout << std::left << std::setw(40) << "benchmark name";
    std::cout << std::left << std::setw(15) << "samples";
    std::cout << std::left << std::setw(15) << "median";
    std::cout << std::left << std::setw(15) << "" << std::endl;
    std::cout << std::left << std::setw(40) << "";
    std::cout << std::left << std::setw(15) << "mean";
    std::cout << std::left << std::setw(15) << "low mean";
    std::cout << std::left << std::setw(15) << "high mean" << std::endl;
    std::cout << std::left << std::setw(40) << "";
    std::cout << std::left << std::setw(15) << "std dev";
    std::cout << std::left << std::setw(15) << "low dtd dev";
    std::cout << std::left << std::setw(15) << "high std dev" << std::endl;
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

void print_benchmark(const std::string& name, int n, int b, int d, benchmark_timer::pre_run_info run_info, benchmarkinfo info) {
    if (name.size() > 37) {
        std::cout << std::left << std::setw(38) << name << std::endl
                  << std::left << std::setw(38) << "";
    } else {
        std::cout << std::left << std::setw(38) << name;
    }
    std::cout << std::right << std::setw(15) << run_info.samples;
    std::cout << std::right << std::setw(15) << suitable_prefix(info.medianval).str() << std::endl;

    std::cout << std::left << std::setw(38) << "    n:" + std::to_string(n) + "    b:" + std::to_string(b) + "    d:" + std::to_string(d);
    std::cout << std::right << std::setw(15) << suitable_prefix(info.meanval).str();
    std::cout << std::right << std::setw(15) << suitable_prefix(info.lowmeanval).str();
    std::cout << std::right << std::setw(15) << suitable_prefix(info.highmeanval).str() << std::endl;
    std::cout << std::right << std::setw(38 + 15) << suitable_prefix(info.stddevval).str();
    std::cout << std::right << std::setw(15) << suitable_prefix(info.lowstddevval).str();
    std::cout << std::right << std::setw(15) << suitable_prefix(info.highstddevval).str() << std::endl
              << std::endl;
}

}  // namespace benchmark_timer
