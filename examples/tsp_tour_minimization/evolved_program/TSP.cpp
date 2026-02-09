// standart imports
#include <string>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <algorithm>

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

// Fast Lin-Kernighan style candidate chain evaluation
bool evaluate_candidate_chain(const Config& config, Context& context, int start_city, int max_chain_length, double& best_gain) {
    std::vector<int> chain = {start_city};
    std::vector<bool> in_chain(config.cities_number, false);
    in_chain[start_city] = true;
    
    double current_gain = 0.0;
    int current = start_city;
    
    for (int step = 0; step < max_chain_length; ++step) {
        int best_next = -1;
        double best_step_gain = -1e9;
        
        // Evaluate top candidates for the current city
        for (int i = 0; i < std::min(12, config.candidates_number); ++i) {
            int candidate = context.candidates[current * config.candidates_number + i];
            if (in_chain[candidate]) continue;
            
            // Calculate potential gain from adding this candidate
            double gain = get_distance_double(config, context, current, context.path[current].next) -
                         get_distance_double(config, context, current, candidate);
            
            if (gain > best_step_gain) {
                best_step_gain = gain;
                best_next = candidate;
            }
        }
        
        if (best_next == -1 || best_step_gain <= 0) break;
        
        chain.push_back(best_next);
        in_chain[best_next] = true;
        current_gain += best_step_gain;
        current = best_next;
        
        // Check if closing the tour gives improvement
        double close_gain = get_distance_double(config, context, chain.back(), chain[0]) -
                           get_distance_double(config, context, chain.back(), context.path[chain.back()].next);
        
        double total_gain = current_gain + close_gain;
        if (total_gain > best_gain) {
            best_gain = total_gain;
            return true; // Found improving move
        }
    }
    
    return false;
}


// Fast greedy initial solution based on nearest neighbor from candidate list
void generate_greedy_solution(const Config& config, Context& context, int iteration) {
    std::vector<bool> visited(config.cities_number, false);
    
    int start_city = (iteration * 13 + iteration * iteration) % config.cities_number;
    context.solution[0] = start_city;
    visited[start_city] = true;
    
    int current = start_city;
    
    for (int i = 1; i < config.cities_number; ++i) {
        int best_next = -1;
        double best_dist = std::numeric_limits<double>::max();
        
        // Consider top candidates for speed
        int limit = std::min(12, config.candidates_number);
        for (int j = 0; j < limit; ++j) {
            int candidate = context.candidates[current * config.candidates_number + j];
            if (!visited[candidate]) {
                double dist = get_distance_double(config, context, current, candidate);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_next = candidate;
                }
            }
        }
        
        // Fallback to nearest unvisited city
        if (best_next == -1) {
            for (int j = 0; j < config.cities_number; ++j) {
                if (!visited[j]) {
                    double dist = get_distance_double(config, context, current, j);
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_next = j;
                    }
                }
            }
        }
        
        context.solution[i] = best_next;
        visited[best_next] = true;
        current = best_next;
    }
}


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


static void shake_from_best(const Config& config, Context& context, int swaps) {
    // If best is not initialized yet, bail out
    if (context.best_path_distance_double == inf_double &&
        context.best_path_distance_int32 == inf_int32 &&
        context.best_path_distance_int64 == inf_int64) {
        return;
    }
    // Start from the current best path
    restore_best_path(config, context);
    convert_path_to_solution(config, context);

    // Apply a few random swaps in solution space, then rebuild path
    for (int s = 0; s < swaps; ++s) {
        int a = get_random_int_by_module(config.cities_number);
        int b = get_random_int_by_module(config.cities_number);
        if (a == b) continue;
        std::swap(context.solution[a], context.solution[b]);
    }
    convert_solution_to_path(config, context);
    calc_and_save_total_distance(config, context);
}

void solve(const Config& config, Context& context) {  // the found solution will be stored in context.solution
    // initialization
    std::chrono::time_point<std::chrono::high_resolution_clock> start_total_time = high_resolution_clock::now(), end_total_time;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time, end_time;

    int max_k_opt_depth = config.max_k_opt_depth;

    for (int i = 1; i < config.restarts_number + 1; ++i) {  // time-capped restarts
        double elapsed = duration_cast<duration<double>>(high_resolution_clock::now() - start_total_time).count();
        if (elapsed >= 159.0) break;  // Extended time cap to use full 160s budget

        if (i % 100 == 0) { std::cout << "# --------- Iteration: " << i << '\n'; }
        int improved_times = 0;

    // Optimized restart strategy with better time-quality tradeoff
    start_time = high_resolution_clock::now();
    double time_elapsed = duration_cast<duration<double>>(high_resolution_clock::now() - start_total_time).count();
    
    // Time-aware restart strategy with progressive intensification
    if (time_elapsed > 40.0 && i > 60) {
        double quality_ratio = context.path_distance_double / context.best_path_distance_double;
        
        // More aggressive exploitation strategy
        if (quality_ratio < 1.006) {
            // Elite solution - intensive local search
            if (i % 3 == 0) {
                shake_from_best(config, context, 1); // Light perturbation
            } else {
                restore_best_path(config, context);
            }
        } else if (quality_ratio < 1.020) {
            // Good solution - balanced approach
            if (i % 4 == 0) {
                shake_from_best(config, context, 3); // Moderate perturbation
            } else {
                generate_greedy_solution(config, context, i);
                convert_solution_to_path(config, context);
            }
        } else {
            // Poor solution - fresh start with greedy
            generate_greedy_solution(config, context, i);
            convert_solution_to_path(config, context);
        }
    } else if (i <= 50 || context.best_path_distance_double == inf_double) {
        // Early phase: rapid exploration
        if (i % 3 == 0) {
            generate_random_solution(config, context);
        } else {
            generate_greedy_solution(config, context, i);
        }
        convert_solution_to_path(config, context);
    } else {
        // Middle phase: balanced exploration
        if (i % 8 == 0) {
            generate_random_solution(config, context);
        } else {
            generate_greedy_solution(config, context, i);
        }
        convert_solution_to_path(config, context);
    }
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

        // Selective 3-opt search for better solution quality
        start_time = high_resolution_clock::now();
        int improved_3_opt_times = 0;
        double current_ratio = context.path_distance_double / context.best_path_distance_double;
        
        // More frequent 3-opt activation for elite solutions (0.4% threshold)
        if (current_ratio < 1.004 && i % 2 == 0) { // Elite solutions, more frequent
            int max_3_opt_iterations = 25;
            
            for (int three_opt_iter = 0; three_opt_iter < max_3_opt_iterations; three_opt_iter++) {
                bool improved = false;
                
                // Spatial distribution of starting points
                int city1 = (three_opt_iter * 7919) % config.cities_number;
                
                // Efficient candidate evaluation
                int candidate_limit = std::min(8, config.candidates_number);
                for (int candidate_idx1 = 0; candidate_idx1 < candidate_limit && !improved; candidate_idx1++) {
                    int city2 = context.candidates[city1 * config.candidates_number + candidate_idx1];
                    if (city1 == city2) continue;
                    
                    int second_level_limit = std::min(6, config.candidates_number);
                    for (int candidate_idx2 = 0; candidate_idx2 < second_level_limit && !improved; candidate_idx2++) {
                        int city3 = context.candidates[city2 * config.candidates_number + candidate_idx2];
                        
                        if (city1 == city3 || city2 == city3) continue;
                        
                        if (apply_3_opt_move(config, context, city1, city2, city3)) {
                            improved_3_opt_times++;
                            improved = true;
                            break;
                        }
                    }
                }
            }
        }
        end_time = high_resolution_clock::now();

        if (config.distance_type != DistanceType::Double) {
            context.path_distance_double = calc_total_distance_double(config, context);
        }
        if (i % 100 == 0) { std::cout << std::setprecision(8) << "Phase #3 (local 3'opt search). Total distance: " << context.path_distance_double << ", Improved times: " << improved_3_opt_times << ", Time: " << duration_cast<milliseconds>(end_time - start_time).count() << " ms\n"; }

        // Selective k-opt activation for better solution quality
        start_time = high_resolution_clock::now();
        current_ratio = context.path_distance_double / context.best_path_distance_double;
        
        int adaptive_k_opt_depth = 0;
        int adaptive_simulations = 0;
        
        // Aggressive k-opt activation for elite solutions (0.1% threshold)
        if (current_ratio < 1.001 && i % 2 == 0) { // Elite solutions, more frequent
            adaptive_k_opt_depth = max_k_opt_depth;
            adaptive_simulations = config.max_k_opt_simulations_without_improve_to_stop;
        } else if (current_ratio < 1.0025 && i % 4 == 0) { // Good solutions, less frequent (0.25% threshold)
            adaptive_k_opt_depth = std::min(24, max_k_opt_depth);
            adaptive_simulations = std::min(20, config.max_k_opt_simulations_without_improve_to_stop);
        }
        
        if (adaptive_k_opt_depth > 0) {
            improved_times = local_k_opt_search(config, context, adaptive_k_opt_depth);
        } else {
            improved_times = 0;
        }
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
        // More aggressive depth adaptation for better solution quality
        double progress = static_cast<double>(i) / config.restarts_number;
        double quality_factor = context.path_distance_double / context.best_path_distance_double;
        double time_elapsed = duration_cast<duration<double>>(high_resolution_clock::now() - start_total_time).count();
        double time_remaining = 159.0 - time_elapsed;
        
        // More aggressive time utilization for solution quality
        double time_factor = std::min(1.0, time_remaining / 60.0);
        
        if (progress < 0.3) {
            // Early phase - deeper exploration
            max_k_opt_depth = 22 + (rand() % (10 + static_cast<int>(8 * time_factor)));
        } else if (progress < 0.7) {
            // Middle phase - balanced depth
            if (quality_factor < 1.006) {
                max_k_opt_depth = 26 + (rand() % (12 + static_cast<int>(6 * time_factor)));
            } else {
                max_k_opt_depth = 24 + (rand() % (10 + static_cast<int>(5 * time_factor)));
            }
        } else {
            // Late phase - focused depth for elite solutions
            if (quality_factor < 1.004 && time_remaining > 30.0) {
                max_k_opt_depth = 28 + (rand() % (10 + static_cast<int>(6 * time_factor)));
            } else {
                max_k_opt_depth = 24 + (rand() % (8 + static_cast<int>(4 * time_factor)));
            }
        }
        max_k_opt_depth = std::min(max_k_opt_depth, config.max_k_opt_depth);
        max_k_opt_depth = std::max(18, max_k_opt_depth);
    }

        if (i % 100 == 0) { std::cout << '\n'; }
	}

    // Final intensification phase
    end_total_time = high_resolution_clock::now();
    double total_elapsed = duration_cast<duration<double>>(end_total_time - start_total_time).count();
    double time_remaining = 159.0 - total_elapsed;
    if (time_remaining > 2.0) {
        // Restore best path to context.path
        restore_best_path(config, context);
        convert_path_to_solution(config, context);
        calc_and_save_total_distance(config, context);
        
        auto intensification_start = high_resolution_clock::now();
        double intensification_time_limit = std::min(5.0, time_remaining - 1.0);
        int intensification_improvements = 0;
        
        // Run 3-opt intensification with a time limit
        bool improved = true;
        while (improved && duration_cast<duration<double>>(high_resolution_clock::now() - intensification_start).count() < intensification_time_limit) {
            improved = false;
            // Try each city as starting point
            for (int city1 = 0; city1 < config.cities_number; ++city1) {
                int limit1 = std::min(12, config.candidates_number);
                for (int idx1 = 0; idx1 < limit1; ++idx1) {
                    int city2 = context.candidates[city1 * config.candidates_number + idx1];
                    if (city1 == city2) continue;
                    int limit2 = std::min(8, config.candidates_number);
                    for (int idx2 = 0; idx2 < limit2; ++idx2) {
                        int city3 = context.candidates[city2 * config.candidates_number + idx2];
                        if (city1 == city3 || city2 == city3) continue;
                        if (apply_3_opt_move(config, context, city1, city2, city3)) {
                            improved = true;
                            ++intensification_improvements;
                            break;
                        }
                    }
                    if (improved) break;
                }
                if (improved) break;
                // Check time
                if (duration_cast<duration<double>>(high_resolution_clock::now() - intensification_start).count() >= intensification_time_limit)
                    break;
            }
        }
        // If improvements found, update best path
        if (intensification_improvements > 0) {
            calc_and_save_total_distance(config, context);
            store_path_as_best(config, context);
        }
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