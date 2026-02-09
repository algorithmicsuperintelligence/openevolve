#pragma once


// * The following functions are used to randomly generate an initial solution.
// * Starting from an arbitrarily chosen city, it iteratively selects a random city until forming a complete TSP tour.


int select_random_city(const Config& config, Context& context) {
	int random_start = get_random_int_by_module(config.cities_number);

	for (int i = 0; i < config.cities_number; ++i) {
		int city = (random_start + i) % config.cities_number;

		if (!context.is_city_selected[city]) {
			return city;
		}
	}

	return null;
}

void generate_random_solution(const Config& config, Context& context) {
	for (int i = 0; i < config.cities_number; ++i) {
		context.solution[i] = null;
		context.is_city_selected[i] = false;
	}

    for (int i = 0; i < config.cities_number; ++i) {
        int current_city = select_random_city(config, context);
        context.solution[i] = current_city;
        context.is_city_selected[current_city] = true;
    }
}