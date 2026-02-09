#pragma once

// standart imports
#include <string>
#include <vector>
#include <limits>
#include <iostream>


// global variables declaration
constexpr int null = -1;
constexpr double e = 2.718281;
constexpr double inf_double = std::numeric_limits<double>::infinity();
constexpr int inf_int32 = std::numeric_limits<int>::max();
constexpr long long inf_int64 = std::numeric_limits<long long>::max();


enum class CandidatesSource { KNN, HeatMap };
enum class DistanceType { Int32, Int64, Double };


struct Config {
    int cities_number;
    std::string input_path;
    std::string output_path;
    bool use_heat_map_as_initial_weights;
    CandidatesSource candidates_source;
    int candidates_number;
    int max_k_opt_depth;
    bool random_k_opt_depth_after_first_iteration;
    double min_potential_to_consider;
    double exploration_coefficient;
    double weight_delta_coefficient;
    bool use_sensitivity_decrease;
    double sensitivity_temperature;
    int max_k_opt_simulations_without_improve_to_stop;
    int restarts_number;
    DistanceType distance_type;
    double magnify_rate;

    Config(const json& config) {
        cities_number = config["cities_number"];
        input_path = config["input_path"];
        output_path = config["output_path"];
        use_heat_map_as_initial_weights = config["use_heat_map_as_initial_weights"];

        if (config["candidates_source"].get<std::string>() == "knn") {
            candidates_source = CandidatesSource::KNN;
        } else if (config["candidates_source"].get<std::string>() == "heat_map") {
            candidates_source = CandidatesSource::HeatMap;
        } else {
            throw std::invalid_argument("Unknown candidates source: '" + config["candidates_source"].get<std::string>() + "'.");
        }

        candidates_number = config["candidates_number"];
        max_k_opt_depth = config["max_k_opt_depth"];
        random_k_opt_depth_after_first_iteration = config["random_k_opt_depth_after_first_iteration"];
        min_potential_to_consider = config["min_potential_to_consider"];
        exploration_coefficient = config["exploration_coefficient"];
        weight_delta_coefficient = config["weight_delta_coefficient"];
        use_sensitivity_decrease = config["use_sensitivity_decrease"];
        sensitivity_temperature = config["sensitivity_temperature"];
        max_k_opt_simulations_without_improve_to_stop = config["max_k_opt_simulations_without_improve_to_stop"];
        restarts_number = config["restarts_number"];

        if (config["distance_type"].get<std::string>() == "int32") {
            distance_type = DistanceType::Int32;
        } else if (config["distance_type"].get<std::string>() == "int64") {
            distance_type = DistanceType::Int64;
        } else if (config["distance_type"].get<std::string>() == "double") {
            distance_type = DistanceType::Double;
        } else {
            throw std::invalid_argument("Unknown distance type: '" + config["distance_type"].get<std::string>() + "'.");
        }

        magnify_rate = config["magnify_rate"];

        // warnings
        if (distance_type == DistanceType::Double) {
            if (cities_number > 1'000'000) {
                std::cout << "Warning: `cities_number` is greater than 1'000'000. Try to use with `distance_type` = 'int64' to avoid precision errors.\n";
            }
        } else {
            if (magnify_rate < 10'000.0) {
                std::cout << "Warning: `magnify_rate` is better to take more than 10'000 when using with `distance_type` = 'int32' or 'int64'.\n";
            }
            if (distance_type == DistanceType::Int32 && static_cast<long long>(cities_number) * magnify_rate > 1'000'000'000) {
                std::cout << "Warning: distance might be out of bounds for a 32 bit integer. Consider switching to doubles for distance calculation or using 64 bit integers.\n";
            }
            if (distance_type == DistanceType::Int64 && magnify_rate > 1'000'000'000) {
                std::cout << "Warning: distance might be out of bounds for a 64 bit integer. Consider switching to doubles for distance calculation.\n";
            }
        }
    }
};


struct City {
  int prev;
  int next;
};


struct Context {
    // coordinates
    double* coordinates_double_x = nullptr;
    double* coordinates_double_y = nullptr;

    int* coordinates_int32_x = nullptr;
    int* coordinates_int32_y = nullptr;

    long long* coordinates_int64_x = nullptr;
    long long* coordinates_int64_y = nullptr;

    // distances
    double* distance_double = nullptr;
    int* distance_int32 = nullptr;
    long long* distance_int64 = nullptr;

    // weights
    double* heat_map = nullptr;
    double* weight = nullptr;
    double* total_weight = nullptr;
    double* potential = nullptr;

    // candidates
    int* candidates = nullptr;

    // local k opt search
    int* pairs = nullptr;
    int* saved_pairs = nullptr;
    int saved_depth = 0;

    double current_best_delta_double = -inf_double;
    int current_best_delta_int32 = -inf_int32;
    long long current_best_delta_int64 = -inf_int64;

    long long* chosen_times = nullptr;
    long long total_simulations = 0;

    // path
    City* path = nullptr;
    double path_distance_double = inf_double;
    int path_distance_int32 = inf_int32;
    long long path_distance_int64 = inf_int64;

    City* best_path = nullptr;
    double best_path_distance_double = inf_double;
    int best_path_distance_int32 = inf_int32;
    long long best_path_distance_int64 = inf_int64;

    int* solution;
    std::vector<bool> is_city_selected;

    // other utils/tmp variables
    std::vector<int> buffer;

    Context(const Config& config) : buffer(config.cities_number, 0), is_city_selected(config.cities_number, false) {
        coordinates_double_x = new double[config.cities_number];
        coordinates_double_y = new double[config.cities_number];

        if (config.distance_type == DistanceType::Int32) {
            coordinates_int32_x = new int[config.cities_number];
            coordinates_int32_y = new int[config.cities_number];
        }
        if (config.distance_type == DistanceType::Int64) {
            coordinates_int64_x = new long long[config.cities_number];
            coordinates_int64_y = new long long[config.cities_number];
        }

        distance_double = new double[config.cities_number * config.cities_number];
        if (config.distance_type == DistanceType::Int32) {
            distance_int32 = new int[config.cities_number * config.cities_number];
        }
        if (config.distance_type == DistanceType::Int64) {
            distance_int64 = new long long[config.cities_number * config.cities_number];
        }

        if (config.use_heat_map_as_initial_weights) {
            heat_map = new double[config.cities_number * config.cities_number];
        }
        weight = new double[config.cities_number * config.cities_number];
        total_weight = new double[config.cities_number];
        potential = new double[config.cities_number * config.cities_number];

        candidates = new int[config.cities_number * config.candidates_number];

        pairs = new int[config.cities_number];  // because currently we have a random k opt depth change option
        saved_pairs = new int[config.cities_number];

        chosen_times = new long long[config.cities_number * config.cities_number];
        for (int i = 0; i < config.cities_number * config.cities_number; ++i) {
            chosen_times[i] = 0;
        }

        path = new City[config.cities_number];
        best_path = new City[config.cities_number];
        solution = new int[config.cities_number];
    }

    ~Context() {
        delete[] coordinates_double_x;
        delete[] coordinates_double_y;

        if (coordinates_int32_x) { delete[] coordinates_int32_x; }
        if (coordinates_int32_y) { delete[] coordinates_int32_y; }

        if (coordinates_int64_x) { delete[] coordinates_int64_x; }
        if (coordinates_int64_y) { delete[] coordinates_int64_y; }

        delete[] distance_double;
        if (distance_int32) { delete[] distance_int32; }
        if (distance_int64) { delete[] distance_int64; }

        if (heat_map) { delete[] heat_map; }
        delete[] weight;
        delete[] total_weight;
        delete[] potential;

        delete[] candidates;

        delete[] pairs;
        delete[] saved_pairs;

        delete[] chosen_times;

        delete[] path;
        delete[] best_path;
        delete[] solution;
    }
};


void convert_solution_to_path(const Config& config, Context& context) {
    for (int i = 0; i < config.cities_number; ++i) {
        int current_city = context.solution[i];

        context.path[current_city].prev = context.solution[(i + config.cities_number - 1) % config.cities_number];
        context.path[current_city].next = context.solution[(i + 1) % config.cities_number];
    }
}

void convert_path_to_solution(const Config& config, Context& context) {
    int current_city = 0;

    for (int i = 0; i < config.cities_number; ++i) {
        context.solution[i] = current_city;

        current_city = context.path[current_city].next;
    }
}

void store_path_as_best(const Config& config, Context& context) {
    for (int i = 0; i < config.cities_number; ++i) {
        context.best_path[i].prev = context.path[i].prev;
        context.best_path[i].next = context.path[i].next;
    }

    if (config.distance_type == DistanceType::Double) {
        context.best_path_distance_double = context.path_distance_double;
    }
    if (config.distance_type == DistanceType::Int32) {
        context.best_path_distance_int32 = context.path_distance_int32;
    }
    if (config.distance_type == DistanceType::Int64) {
        context.best_path_distance_int64 = context.path_distance_int64;
    }
}

void restore_best_path(const Config& config, Context& context) {
    for (int i = 0; i < config.cities_number; ++i) {
        context.path[i].prev = context.best_path[i].prev;
        context.path[i].next = context.best_path[i].next;
    }
}
