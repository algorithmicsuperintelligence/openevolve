#pragma once

// standart imports
#include <cmath>
#include <iostream>


void calc_potential_inplace(const Config& config, Context& context, int i, int j) {
    if (i == j) {
        context.potential[i * config.cities_number + j] = 0.0;
        return;
    }

    // calculating total weight
    double average_weight = context.total_weight[i] / (config.cities_number - 1);
    double weight = context.weight[i * config.cities_number + j];

    context.potential[i * config.cities_number + j] = (smooth_relu(weight) / average_weight) + config.exploration_coefficient * sqrt(log(context.total_simulations + 1) / (context.chosen_times[i * config.cities_number + j] + 1));  // always a positive value
}

int get_candidate_proportionally_by_potential(const Config& config, Context& context, int current_city, int start_city) {
    int next_city = context.path[current_city].next;

	double total_potential = 0.0;
    int candidates_available = 0;

	for (int i = 0; i < config.candidates_number; ++i) {
        int candidate = context.candidates[current_city * config.candidates_number + i];

        calc_potential_inplace(config, context, current_city, candidate);
        double potential = context.potential[current_city * config.cities_number + candidate];

        if (candidate == next_city || candidate == start_city || (potential < config.min_potential_to_consider)) { continue; }  // not available

		total_potential += potential;
        ++candidates_available;
	}

    if (candidates_available == 0) { return null; }

    // choosing the random available candidate proportionally
    double random_potential = (static_cast<double>(rand()) / RAND_MAX) * total_potential;

	for (int i = 0; i < config.candidates_number; ++i) {
        int candidate = context.candidates[current_city * config.candidates_number + i];
        double potential = context.potential[current_city * config.cities_number + candidate];

        if (candidate == next_city || candidate == start_city || (potential < config.min_potential_to_consider)) { continue; }  // not available

        random_potential -= potential;
        if (random_potential <= 0) { return candidate; }
    }

	return null;
}


bool apply_k_opt_move(const Config& config, Context& context, int start_city, int max_k_opt_depth) {
    ++context.total_simulations;

    // first pair
    int next_to_start_city = context.path[start_city].next;

    context.pairs[0] = start_city;
    context.pairs[1] = next_to_start_city;
    int depth = 1;

    int copied_to_saved_pairs = 0;
    bool need_copy_to_saved_pairs = false;

    // breaking an edge of the first pair
    context.path[start_city].next = null;
    context.path[next_to_start_city].prev = null;

    // initializing gains
    double gain_double; double gain_double_with_closure = 0.0;
    int gain_int32; int gain_int32_with_closure = 0;
    long long gain_int64; long long gain_int64_with_closure = 0;

    if (config.distance_type == DistanceType::Double) {
        gain_double = get_distance_double(config, context, start_city, next_to_start_city);
    }
    if (config.distance_type == DistanceType::Int32) {
        gain_int32 = get_distance_int32(config, context, start_city, next_to_start_city);
    }
    if (config.distance_type == DistanceType::Int64) {
        gain_int64 = get_distance_int64(config, context, start_city, next_to_start_city);
    }

    bool apply_move = false;

    int current_city = next_to_start_city;

    for (int i = 1; i < max_k_opt_depth; ++i) {
        int proposed_city = get_candidate_proportionally_by_potential(config, context, current_city, start_city);

        if (proposed_city == null) { return false; }  // no candidates, could not improve

        ++context.chosen_times[current_city * config.cities_number + proposed_city];
		++context.chosen_times[proposed_city * config.cities_number + current_city];

        int proposed_city_link = context.path[proposed_city].prev;  // city to disconnect from the proposed city (and maybe to connect to the start city)

        context.pairs[2 * i] = proposed_city;
        context.pairs[2 * i + 1] = proposed_city_link;
        ++depth;

        // applying 2 opt move
        reverse_sub_path(context, current_city, proposed_city_link);

        context.path[current_city].next = proposed_city;
        context.path[proposed_city].prev = current_city;
        context.path[proposed_city_link].prev = null;

        // recalculating gains
        if (config.distance_type == DistanceType::Double) {
            gain_double += get_distance_double(config, context, proposed_city_link, proposed_city) - get_distance_double(config, context, current_city, proposed_city);
            gain_double_with_closure = gain_double - get_distance_double(config, context, start_city, proposed_city_link);

            if (gain_double_with_closure > 0.0) {
                apply_move = true;
                context.path_distance_double -= gain_double_with_closure;
            }
            if (context.current_best_delta_double < gain_double_with_closure) {
                need_copy_to_saved_pairs = true;
                context.current_best_delta_double = gain_double_with_closure;
            }
        }
        if (config.distance_type == DistanceType::Int32) {
            gain_int32 += get_distance_int32(config, context, proposed_city_link, proposed_city) - get_distance_int32(config, context, current_city, proposed_city);
            gain_int32_with_closure = gain_int32 - get_distance_int32(config, context, start_city, proposed_city_link);

            if (gain_int32_with_closure > 0) {
                apply_move = true;
                context.path_distance_int32 -= gain_int32_with_closure;
            }
            if (context.current_best_delta_int32 < gain_int32_with_closure) {
                need_copy_to_saved_pairs = true;
                context.current_best_delta_int32 = gain_int32_with_closure;
            }
        }
        if (config.distance_type == DistanceType::Int64) {
            gain_int64 += get_distance_int64(config, context, proposed_city_link, proposed_city) - get_distance_int64(config, context, current_city, proposed_city);
            gain_int64_with_closure = gain_int64 - get_distance_int64(config, context, start_city, proposed_city_link);

            if (gain_int64_with_closure > 0) {
                apply_move = true;
                context.path_distance_int64 -= gain_int64_with_closure;
            }
            if (context.current_best_delta_int64 < gain_int64_with_closure) {
                need_copy_to_saved_pairs = true;
                context.current_best_delta_int64 = gain_int64_with_closure;
            }
        }

        if (need_copy_to_saved_pairs) {  // for future weight updating
            for (int i = copied_to_saved_pairs; i < depth; ++i) {
                context.saved_pairs[2 * i] = context.pairs[2 * i];
                context.saved_pairs[2 * i + 1] = context.pairs[2 * i + 1];
            }

            copied_to_saved_pairs = depth;
            context.saved_depth = depth;

            need_copy_to_saved_pairs = false;
        }

        if (apply_move) { break; }

        current_city = proposed_city_link;
    }

    if (apply_move) {
        // concluding the path to cycle
        int end_city = context.pairs[2 * depth - 1];

        context.path[start_city].next = end_city;
        context.path[end_city].prev = start_city;

        return true;
    }
    return false;
}


bool improve_by_k_opt_move(const Config& config, Context& context, int max_k_opt_depth) {
    context.current_best_delta_double = -inf_double;
    context.current_best_delta_int32 = -inf_int32;
    context.current_best_delta_int64 = -inf_int64;

    // saving current path length
    double saved_path_distance_double = context.path_distance_double;
    double saved_path_distance_int32 = context.path_distance_int32;
    double saved_path_distance_int64 = context.path_distance_int64;

    bool improved = false;

    for (int i = 0; i < config.max_k_opt_simulations_without_improve_to_stop; ++i) {
        // saving current path
        convert_path_to_solution(config, context);

        int start_city = get_random_int_by_module(config.cities_number);
        if (apply_k_opt_move(config, context, start_city, max_k_opt_depth)) {
            improved = true;
            break;
        }

        // restoring the path that was before the move
        convert_solution_to_path(config, context);
    }

    // updating weights from the best delta (it can be negative, if we've not improved)
    double weight_delta = 0.0;
    if (config.distance_type == DistanceType::Double && context.current_best_delta_double != -inf_double) {
        weight_delta = config.weight_delta_coefficient * (pow(e, context.current_best_delta_double / saved_path_distance_double) - 1);
    }
    if (config.distance_type == DistanceType::Int32 && context.current_best_delta_int32 != -inf_int32) {
        weight_delta = config.weight_delta_coefficient * (pow(e, static_cast<double>(context.current_best_delta_int32) / saved_path_distance_int32) - 1);
    }
    if (config.distance_type == DistanceType::Int64 && context.current_best_delta_int64 != -inf_int64) {
        weight_delta = config.weight_delta_coefficient * (pow(e, static_cast<double>(context.current_best_delta_int64) / saved_path_distance_int64) - 1);
    }

    for (int i = 0; i < context.saved_depth; ++i) {
        int current_city = context.saved_pairs[2 * i];
        int proposed_city = (i < context.saved_depth - 1) ? context.saved_pairs[2 * i + 2] : context.saved_pairs[0];

        double factor = 1.0;
        if (config.use_sensitivity_decrease && !improved) {
            // exponential decrease in sensitivity
            factor = pow(e, -i / config.sensitivity_temperature);
        }

        update_weight_undirected(config, context, current_city, proposed_city, weight_delta * factor);
    }

    return improved;
}

int local_k_opt_search(const Config& config, Context& context, int max_k_opt_depth) {  
    // calculating total weight (for mitigating precision based errors, that was found empirically)
    for (int i = 0; i < config.cities_number; ++i) {
        double total_weight = 0.0;

        for (int j = 0; j < config.cities_number; ++j) {
            total_weight += smooth_relu(context.weight[i * config.cities_number + j]);
        }

        context.total_weight[i] = total_weight;
    }

    // running simulations and trying to improve
    int improved_times = 0;

    while (improve_by_k_opt_move(config, context, max_k_opt_depth)) { ++improved_times; };

    return improved_times;
}
