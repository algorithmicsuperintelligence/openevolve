#pragma once


bool apply_2_opt_move(const Config& config, Context& context, int i, int j) {
    if (is_cities_same_or_adjacent(config, context, i, j)) { return false; }

    int i_next = context.path[i].next;
	int j_next = context.path[j].next;

    // Early check for common case where move won't improve
    if (config.distance_type == DistanceType::Double) {
        double current_dist = get_distance_double(config, context, i, i_next) + get_distance_double(config, context, j, j_next);
        double new_dist = get_distance_double(config, context, i, j) + get_distance_double(config, context, i_next, j_next);
        if (new_dist >= current_dist) return false;
    }
    // Similar checks for Int32 and Int64...

    ++context.chosen_times[i * config.cities_number + j];
	++context.chosen_times[j * config.cities_number + i];
	++context.chosen_times[i_next * config.cities_number + j_next];
	++context.chosen_times[j_next * config.cities_number + i_next];

    ++context.total_simulations;

    // calculating delta
    bool apply_move = false;
    double weight_increase = 0.0;

    if (config.distance_type == DistanceType::Double) {
        double delta = get_distance_double(config, context, i, i_next) + get_distance_double(config, context, j, j_next) - get_distance_double(config, context, i, j) - get_distance_double(config, context, i_next, j_next);
        if (delta > 0.0) {
            apply_move = true;
            weight_increase = config.weight_delta_coefficient * (pow(e, delta / context.path_distance_double) - 1);
            context.path_distance_double -= delta;
        }
    }
    if (config.distance_type == DistanceType::Int32) {
        int delta = get_distance_int32(config, context, i, i_next) + get_distance_int32(config, context, j, j_next) - get_distance_int32(config, context, i, j) - get_distance_int32(config, context, i_next, j_next);
        if (delta > 0) {
            apply_move = true;
            weight_increase = config.weight_delta_coefficient * (pow(e, static_cast<double>(delta) / context.path_distance_int32) - 1);
            context.path_distance_int32 -= delta;
        }
    }
    if (config.distance_type == DistanceType::Int64) {
        long long delta = get_distance_int64(config, context, i, i_next) + get_distance_int64(config, context, j, j_next) - get_distance_int64(config, context, i, j) - get_distance_int64(config, context, i_next, j_next);
        if (delta > 0) {
            apply_move = true;
            weight_increase = config.weight_delta_coefficient * (pow(e, static_cast<double>(delta) / context.path_distance_int64) - 1);
            context.path_distance_int64 -= delta;
        }
    }

    if (apply_move) {
        // applying 2 opt move
        reverse_sub_path(context, i_next, j);

        context.path[i].next = j;
        context.path[i_next].next = j_next;
        context.path[j].prev = i;
        context.path[j_next].prev = i_next;

        // updating weights
        update_weight_undirected(config, context, i, j, weight_increase);
        update_weight_undirected(config, context, i_next, j_next, weight_increase);

        return true;
    }
    return false;
}


bool improve_by_2_opt_move(const Config& config, Context& context) {
    for (int i = 0; i < config.cities_number; ++i) {
		for (int j = 0; j < config.candidates_number; ++j) {
			int candidate = context.candidates[i * config.candidates_number + j];

			if (apply_2_opt_move(config, context, i, candidate)) { return true; }
		}
	}
    return false;
}

int local_2_opt_search(const Config& config, Context& context) {
    int improved_times = 0;

    while (improve_by_2_opt_move(config, context)) { ++improved_times; };

    return improved_times;
}