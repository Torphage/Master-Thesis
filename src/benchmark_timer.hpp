#pragma once
#ifndef BENCHMARK_TIMER_HPP
#define BENCHMARK_TIMER_HPP

#include "../benchmark/benchmark_json.hpp"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <string>
#include <vector>
#include <iostream>

namespace benchmark_timer {

void print_header();
void print_benchmark(const std::string& name, int n, int b, int d, benchmark_json::config_information& config_info);

template <typename Word>
double mean(std::vector<Word> const& v) {
    if (v.empty()) {
        return 0;
    }

    auto const count = static_cast<double>(v.size());

    return std::reduce(v.begin(), v.end()) / count;
}

template <typename Word>
double low_mean(std::vector<Word>& v) {
    std::vector<double> temp = v;
    if (temp.empty()) {
        return 0;
    }

    auto const middle = temp.begin() + temp.size() / 2;
    std::nth_element(temp.begin(), middle, temp.end());
    auto const count = static_cast<double>(middle - temp.begin());

    return std::reduce(temp.begin(), middle) / count;
}

template <typename Word>
double high_mean(std::vector<Word>& v) {
    std::vector<double> temp = v;
    if (temp.empty()) {
        return 0;
    }

    auto const middle = temp.begin() + temp.size() / 2;
    std::nth_element(temp.begin(), middle, temp.end());
    auto const count = static_cast<double>(temp.end() - middle);

    return std::reduce(middle, temp.end()) / count;
}

template <typename Word>
double median(std::vector<Word>& v) {
    std::vector<double> temp = v;
    int n = static_cast<int>(temp.size());
    std::nth_element(temp.begin(), temp.begin() + n / 2, temp.end());
    double median1 = temp[n / 2];

    if (n % 2 != 0) {
        return median1;
    } else {
        std::nth_element(temp.begin(), temp.begin() + (n - 1) / 2, temp.end());
        double median2 = temp[n / 2 - 1];
        return (median1 + median2) / 2.0;
    }
}

template <typename Word>
double variance(std::vector<Word> const& v) {
    int n = static_cast<int>(v.size());
    double var = 0;
    double m = mean(v);
    for (int i = 0; i < n; i++) {
        var += (v[i] - m) * (v[i] - m);
    }
    var /= n;
    return var;
}

template <typename Word>
double stddev(std::vector<Word> const& v) {
    return sqrt(variance(v));
}

template <class Lambda>
static double time(Lambda&& fn) {
    auto start = std::chrono::steady_clock::now();

    fn();

    auto finish = std::chrono::steady_clock::now();

    return std::chrono::duration_cast<
               std::chrono::duration<double>>(finish - start)
        .count();
}

template <class Lambda, class... Args>
static double time(Lambda&& fn, Args&&... args) {
    return time(
        [=]() mutable { return fn(args...); });
}

template <class Lambda, class... Args>
void benchmark(benchmark_json::config_information& config_info, Lambda&& fn, Args&&... args) {
    int i = config_info.warmup_iterations;
    while (i--) {
        time(fn, args...);
    }

    std::vector<double> vec(config_info.samples);
    int j = config_info.samples;
    while (j) {
        vec[config_info.samples - j] = time(fn, args...);
        j--;
    }

    // std::cout << config_info.samples << config_info.function << std::endl;

    config_info.results.vals = vec;
    config_info.results.mean_val = mean(vec);
    config_info.results.low_mean_val = low_mean(vec);
    config_info.results.high_mean_val = high_mean(vec);
    config_info.results.median_val = median(vec);
    config_info.results.variance_val = variance(vec);
    config_info.results.std_dev_val = stddev(vec);

    return;
}

}  // namespace benchmark_timer

#endif
