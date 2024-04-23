#pragma once
#ifndef BENCHMARK_TIMER_HPP
#define BENCHMARK_TIMER_HPP

#include "../benchmark/benchmark_json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>
#include <iostream>

namespace benchmark_timer {

void print_header();
void print_pre_run_info(benchmark_json::config_information& config_info, const double time);
void print_benchmark(benchmark_json::config_information& config_info);

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
    std::vector<double> warmup_vec(config_info.warmup_iterations);
    std::vector<double> vec(config_info.samples);

    int estimation_samples = static_cast<int>(std::min(100.0, std::ceil(config_info.warmup_iterations / 20.0)));
    int i = 0;

    // Benchmark only a few times, depending on the number of warmup iterations,
    // this will give the user an estimation for how long each sample take. 
    while (i < estimation_samples) {
        warmup_vec[i] = time(fn, args...);
        i++;
    }

    // Print the name of what is being benchmarked, how many samples it takes,
    // as well as the estimated run time for each sample.
    print_pre_run_info(config_info, (config_info.warmup_iterations + config_info.samples - estimation_samples) * mean(std::vector<double>(warmup_vec.begin(), warmup_vec.begin() + estimation_samples)));

    // Perform the rest of the warmup samples, and store them in a separate vector,
    // such that they can be viewed in the output json file.
    while (i < config_info.warmup_iterations) {
        warmup_vec[i] = time(fn, args...);
        i++;
    }

    // Benchmark the program and store it in a new vector
    int j = 0;
    while (j < config_info.samples) {
        vec[j] = time(fn, args...);
        j++;
    }

    config_info.results.warmup_vals = warmup_vec;
    config_info.results.vals = vec;
    config_info.results.mean_val = mean(vec);
    config_info.results.low_mean_val = low_mean(vec);
    config_info.results.high_mean_val = high_mean(vec);
    config_info.results.median_val = median(vec);
    config_info.results.variance_val = variance(vec);
    config_info.results.std_dev_val = stddev(vec);

    print_benchmark(config_info);

    return;
}

}  // namespace benchmark_timer

#endif
