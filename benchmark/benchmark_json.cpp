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
        config["name"] = c.name;
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
        config["cores"] = c.cores;
        config["slurm_file"] = c.slurm_file;

        benchmark["config"] = config;

        nlohmann::json result;

        result["mean"] = c.results.mean_val;
        result["low_mean"] = c.results.low_mean_val;
        result["high_mean"] = c.results.high_mean_val;
        result["median"] = c.results.median_val;
        result["variance"] = c.results.variance_val;
        result["stdval"] = c.results.std_dev_val;
        result["geo_mean"] = c.results.geo_mean_val;
        result["min"] = c.results.min_val;
        result["max"] = c.results.max_val;

        nlohmann::json warmups;
        for (const double &time : c.results.warmup_vals) {
            warmups.push_back(time);
        }
        nlohmann::json runs;
        for (const double &time : c.results.vals) {
            runs.push_back(time);
        }

        result["warmups"] = warmups;
        result["timers"] = runs;

        benchmark["result"] = result;

        data["benchmarks"].push_back(benchmark);
    }

    std::ofstream f(filename);
    f << std::setw(4) << data << std::endl;

    return;
}

}  // namespace benchmark_json