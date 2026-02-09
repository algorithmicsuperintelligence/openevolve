// standart imports
#include <string>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

// json imports
#include "include/json.hpp"

using json = nlohmann::json;
using namespace std::chrono;

// other imports
#include "include/context.hpp"
#include "include/utils.hpp"
#include "include/random_solution.hpp"
#include "include/local_2_opt_search.hpp"
#include "include/local_k_opt_search.hpp"

// additional functions & methods
#include "include/additional.hpp"


// --- config parameters ---
//   `cities_number`: number of points on the 2D surface.
//   `input_path`: path to the file with cities coordinates and the edge heat map.
//   `output_path`: path to the file where to write the optimal hamiltonian cycle and corresponding metrics.
//   `use_heat_map_as_initial_weights`: whether to use the heat map as initial for the weights matrix.
//   `candidates_source`: 'knn' or 'heat_map', if 'heat_map' the candidates for each city are the nearest cities to it.
//   `candidates_number`: number of candidate cities for each city.
//   `max_k_opt_depth`: maximum chain links (k parameter) in simulation.
//   `random_k_opt_depth_after_first_iteration`: if to randomly change the `max_k_opt_depth` after the first iteration (restart).
//   `min_potential_to_consider`: minimum potential of an edge to consider it in simulation (look at the formula for potential to understand fully).
//   `exploration_coefficient`: hyperparameter for exploration.
//   `weight_delta_coefficient`: hyperparameter for updating the weights matrix.
//   `use_sensitivity_decrease`: whether to reduce weight flow for deep edges in unsuccessful k'opt search.
//   `sensitivity_temperature`: hyperparameter for controlling the weight decrease in unsuccessful simulation depending on the length of a chain.
//   `max_k_opt_simulations_without_improve_to_stop`: the number of MCTS simulations per restart.
//   `restarts_number`: number of times algorithm restarts while maintaining the weights matrix (number of iterations).
//   `distance_type`: "int32", "int64" or "double".
//   `magnify_rate`: when `distance_type` is "int32" or "int64" algorithm relies only on integers to find an optimal solution (for speed), therefore each distance is magnified by `magnify_rate` value and rounded to integer. (if `distance_type` = 'double' this parameter is ignored)


void read_input_data(const Config& config, Context& context) {
    std::ifstream input_file(config.input_path);

    int cities_number; input_file >> cities_number;

    // reading coordinates
    for (int i = 0; i < config.cities_number; ++i) {
        input_file >> context.coordinates_double_x[i] >> context.coordinates_double_y[i];

        if (config.distance_type == DistanceType::Int32) {
            context.coordinates_int32_x[i] = static_cast<int>(0.5 + context.coordinates_double_x[i] * config.magnify_rate);
            context.coordinates_int32_y[i] = static_cast<int>(0.5 + context.coordinates_double_y[i] * config.magnify_rate);
        }
        if (config.distance_type == DistanceType::Int64) {
            context.coordinates_int64_x[i] = static_cast<long long>(0.5 + context.coordinates_double_x[i] * config.magnify_rate);
            context.coordinates_int64_y[i] = static_cast<long long>(0.5 + context.coordinates_double_y[i] * config.magnify_rate);
        }
    }

    // calculating distances
    for (int i = 0; i < config.cities_number; ++i) {
        for (int j = 0; j < config.cities_number; ++j) {
            context.distance_double[i * config.cities_number + j] = calc_distance_double(context, i, j);
            if (config.distance_type == DistanceType::Int32) {
                context.distance_int32[i * config.cities_number + j] = calc_distance_int32(context, i, j);
            }
            if (config.distance_type == DistanceType::Int64) {
                context.distance_int64[i * config.cities_number + j] = calc_distance_int64(context, i, j);
            }
        }
    }

    if (config.use_heat_map_as_initial_weights) {
        // reading heat map
        for (int i = 0; i < config.cities_number * config.cities_number; ++i) {
            input_file >> context.heat_map[i];
            context.weight[i] = context.heat_map[i];
        }
    } else {
        for (int i = 0; i < config.cities_number * config.cities_number; ++i) {
            context.weight[i] = 0.0;
        }
    }

    // initializing total weight
    for (int i = 0; i < config.cities_number; ++i) {
        double total_weight = 0.0;

        for (int j = 0; j < config.cities_number; ++j) {
            total_weight += smooth_relu(context.weight[i * config.cities_number + j]);
        }

        context.total_weight[i] = total_weight;
    }

    // calculating candidates
    if (config.candidates_source == CandidatesSource::KNN) {
        identify_candidates_for_each_node(config, context, context.distance_double, false);
    } else if (config.candidates_source == CandidatesSource::HeatMap) {
        identify_candidates_for_each_node(config, context, context.heat_map, true);
    } else {
        throw std::invalid_argument("Unknown candidates source.");
    }
}


void solve(const Config& config, Context& context) {  // the found solution will be stored in context.solution
    // initialization
    std::chrono::time_point<std::chrono::high_resolution_clock> start_total_time = high_resolution_clock::now(), end_total_time;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time, end_time;

    int max_k_opt_depth = config.max_k_opt_depth;

	for (int i = 1; i < config.restarts_number + 1; ++i) {  // maybe add some stopping criteria (with BHH 2D constant for example)
        if (i % 100 == 0) { std::cout << "# --------- Iteration: " << i << '\n'; }
        int improved_times = 0;

        // random solution
        start_time = high_resolution_clock::now();
        generate_random_solution(config, context);
        convert_solution_to_path(config, context);
        end_time = high_resolution_clock::now();

        calc_and_save_total_distance(config, context);
        if (config.distance_type != DistanceType::Double) {
            context.path_distance_double = calc_total_distance_double(config, context);
        }
        if (i % 100 == 0) { std::cout << std::setprecision(8) << "Phase #1 (random cycle). Total distance: " << context.path_distance_double << ", Time: " << duration_cast<milliseconds>(end_time - start_time).count() << " ms\n"; }

        // local 2opt search
        start_time = high_resolution_clock::now();
		improved_times = local_2_opt_search(config, context);
        end_time = high_resolution_clock::now();

        if (config.distance_type != DistanceType::Double) {
            context.path_distance_double = calc_total_distance_double(config, context);
        }
        if (i % 100 == 0) { std::cout << std::setprecision(8) << "Phase #2 (local 2'opt search). Total distance: " << context.path_distance_double << ", Improved times: " << improved_times << ", Time: " << duration_cast<milliseconds>(end_time - start_time).count() << " ms\n"; }

        // local k opt search
        start_time = high_resolution_clock::now();
		improved_times = local_k_opt_search(config, context, max_k_opt_depth);
        end_time = high_resolution_clock::now();

        if (config.distance_type != DistanceType::Double) {
            context.path_distance_double = calc_total_distance_double(config, context);
        }
        if (i % 100 == 0) { std::cout << std::setprecision(8) << "Phase #3 (local k'opt search). Total distance: " << context.path_distance_double << ", Improved times: " << improved_times << ", Time: " << duration_cast<milliseconds>(end_time - start_time).count() << " ms\n"; }

        // changing the best path
        if (
            (config.distance_type == DistanceType::Double && context.path_distance_double < context.best_path_distance_double) ||
            (config.distance_type == DistanceType::Int32 && context.path_distance_int32 < context.best_path_distance_int32) ||
            (config.distance_type == DistanceType::Int64 && context.path_distance_int64 < context.best_path_distance_int64)
        ) {
            store_path_as_best(config, context);  // also updates best path distance
        }

        if (config.random_k_opt_depth_after_first_iteration) {
            // random MCTS depth change
            max_k_opt_depth = std::min(10 + (rand() % 80), config.cities_number / 2);
        }

        if (i % 100 == 0) { std::cout << '\n'; }
	}

    // final convertation (context.best_path to context.solution)
    restore_best_path(config, context);
    convert_path_to_solution(config, context);

    end_total_time = high_resolution_clock::now();
    std::cout << "Total elapsed time: " << static_cast<double>(duration_cast<milliseconds>(end_total_time - start_total_time).count()) / 1000 << " sec\n\n";
}


int main(int argc, char** argv) {
    // reading configuration
    std::cout << "Reading configuration...\n";
    if (argc != 2) {
        std::cerr << "Usage: The first and only argument should be the path to the config file.";
        return 1;
    }
    
    std::ifstream config_file(argv[1]);
    json config_raw; config_file >> config_raw;
    Config config(config_raw);

    std::cout << "Number of cities: " << config.cities_number << "\n\n";

    // initialization & memory allocation
    Context context(config);

    // reading input data
    std::cout << "Reading input data...\n";
    read_input_data(config, context);

    // solving
    std::cout << "Solving...\n";
    solve(config, context);

    // printing the solution
    std::ofstream output_file(config.output_path);

    std::cout << "Final solution:\n";
    for (int i = 0; i < config.cities_number; ++i) {
        std::cout << context.solution[i] << ' ';
        output_file << context.solution[i] << ' ';
    }
    std::cout << "\n\nFinal solution score: " << calc_total_distance_double(config, context) << '\n';

    return 0;
}
