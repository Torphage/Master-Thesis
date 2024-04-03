#pragma once
#ifndef BENCHMARK_TIMER_HPP
#define BENCHMARK_TIMER_HPP

#include <algorithm>
#include <chrono>
#include <numeric>
#include <string>
#include <vector>
#include <iostream>

namespace benchmark_timer {

void print_header();
void print_benchmark(const std::string& name, int n, int b, int d, int samples, double median, double mean, double lowMean, double highMean, double stdDev, double lowStdDev, double highStdDev);

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

template <typename Word>
double low_stddev(std::vector<Word>& v) {
    std::vector<double> temp = v;
    std::sort(temp.begin(), temp.end());
    return sqrt(variance(std::vector<Word>(temp.begin(), temp.begin() + temp.size() / 2)));
}

template <typename Word>
double high_stddev(std::vector<Word> const& v) {
    std::vector<double> temp = v;
    std::sort(temp.begin(), temp.end());
    return sqrt(variance(std::vector<Word>(temp.begin() + temp.size() / 2, temp.end())));
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
        [=]() mutable { fn(args...); });
}

template <class Lambda, class... Args>
static benchmarkinfo benchmark(Lambda&& fn, Args&&... args) {
    int repetitions = 100;
    int i = repetitions;
    std::vector<double> vec(repetitions);

    while (i--) {
        vec[repetitions - i] = time(fn, args...);
    }

    // std::cout << median(vec) << std::endl;
    return {
        vec,
        mean(vec),
        low_mean(vec),
        high_mean(vec),
        median(vec),
        variance(vec),
        stddev(vec),
        low_stddev(vec),
        high_stddev(vec),
    };
}

}  // namespace benchmark_timer

#endif
