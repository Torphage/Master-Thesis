#include "benchmark_json.hpp"

#include <fstream>
#include <string>

namespace benchmark_json {

nlohmann::json read_file(std::string filename) {
    std::ifstream f(filename);
    nlohmann::json data;
    f >> data;
    return data;
}

void write_file(std::string filename, information info) {
    nlohmann::json data;
    data["hardware"] = {};
    data["benchmarks"] = nlohmann::json::array();
    for (const config_information &c : info.config) {
        nlohmann::json benchmark = {};

        nlohmann::json config = {};
        config["function"] = c.function;
        config["hash"] = c.hash;
        config["n"] = c.n;
        config["b"] = c.b;
        config["d"] = c.d;
        config["density"] = c.density;
        config["matrix_seed"] = c.matrix_seed;
        config["hash_seed"] = c.hash_seed;
        config["samples"] = c.samples;
        config["warmup_iterations"] = c.warmup_iterations;

        benchmark["config"] = config;

        nlohmann::json result;
        nlohmann::json runs;

        result["mean"] = c.results.mean_val;
        result["low_mean"] = c.results.low_mean_val;
        result["high_mean"] = c.results.high_mean_val;
        result["median"] = c.results.median_val;
        result["variance"] = c.results.variance_val;
        result["stdval"] = c.results.std_dev_val;

        for (const double &time : c.results.vals) {
            runs.push_back(time);
        }

        result["timers"] = runs;

        benchmark["result"] = result;

        data["benchmarks"].push_back(benchmark);
    }

    std::ofstream f(filename);
    f << std::setw(4) << data << std::endl;

    return;
}

}  // namespace benchmark_json