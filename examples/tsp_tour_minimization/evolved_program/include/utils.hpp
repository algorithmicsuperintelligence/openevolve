#pragma once

// standart imports
#include <queue>
#include <cmath>
#include <numeric>
#include <algorithm>

// other imports
#include "context.hpp"


constexpr double BHH_CONSTANT_2D = 0.7120;  // Beardwood–Halton–Hammersley (BHH) constant


long long int64_sqrt(long long value) {
    if (value < 0) return 0;  // invalid for negatives
    if (value < 2) return value;

    constexpr long long MAX_SQRT_LL = 3037000499LL;

    long long left = 1;
    long long right = std::min<long long>(value, MAX_SQRT_LL);
    long long floor_root = 1;

    while (left <= right) {
        long long candidate = left + (right - left) / 2;

        if (candidate <= value / candidate) {
            floor_root = candidate;
            left = candidate + 1;
        } else {
            right = candidate - 1;
        }
    }
    return floor_root;
}

double smooth_relu(double x) {
    if (x < 0) { return pow(e, x); }
    return x + 1.0;
}


double calc_distance_double(Context& context, int i, int j) {
    if (i == j) { return inf_double; }
    double diff_x = (context.coordinates_double_x[i] - context.coordinates_double_x[j]);
    double diff_y = (context.coordinates_double_y[i] - context.coordinates_double_y[j]);
    return sqrt(diff_x * diff_x + diff_y * diff_y);
}

int calc_distance_int32(Context& context, int i, int j) {
    if (i == j) { return inf_int32; }
    long long diff_x = static_cast<long long>(context.coordinates_int32_x[i] - context.coordinates_int32_x[j]);
    long long diff_y = static_cast<long long>(context.coordinates_int32_y[i] - context.coordinates_int32_y[j]);
    return static_cast<int>(int64_sqrt(diff_x * diff_x + diff_y * diff_y));
}

long long calc_distance_int64(Context& context, int i, int j) {
    if (i == j) { return inf_int64; }
    long long diff_x = context.coordinates_int64_x[i] - context.coordinates_int64_x[j];
    long long diff_y = context.coordinates_int64_y[i] - context.coordinates_int64_y[j];
    return int64_sqrt(diff_x * diff_x + diff_y * diff_y);
}


double get_distance_double(const Config& config, Context& context, int i, int j) {
    return context.distance_double[i * config.cities_number + j];
}

int get_distance_int32(const Config& config, Context& context, int i, int j) {
    return context.distance_int32[i * config.cities_number + j];
}

long long get_distance_int64(const Config& config, Context& context, int i, int j) {
    return context.distance_int64[i * config.cities_number + j];
}


double calc_total_distance_double(const Config& config, Context& context) {
    double total_distance = 0.0;

    for (int i = 0; i < config.cities_number; ++i) {
        total_distance += get_distance_double(config, context, i, context.path[i].next);
    }

    return total_distance;
}

int calc_total_distance_int32(const Config& config, Context& context) {
    int total_distance = 0;

    for (int i = 0; i < config.cities_number; ++i) {
        total_distance += get_distance_int32(config, context, i, context.path[i].next);
    }

    return total_distance;
}

long long calc_total_distance_int64(const Config& config, Context& context) {
    long long total_distance = 0;

    for (int i = 0; i < config.cities_number; ++i) {
        total_distance += get_distance_int64(config, context, i, context.path[i].next);
    }

    return total_distance;
}


void calc_and_save_total_distance(const Config& config, Context& context) {
    if (config.distance_type == DistanceType::Double) {
        context.path_distance_double = calc_total_distance_double(config, context);
    }
    if (config.distance_type == DistanceType::Int32) {
        context.path_distance_int32 = calc_total_distance_int32(config, context);
    }
    if (config.distance_type == DistanceType::Int64) {
        context.path_distance_int64 = calc_total_distance_int64(config, context);
    }
}


void update_weight_undirected(const Config& config, Context& context, int i, int j, double weight_delta) {
    context.total_weight[i] -= smooth_relu(context.weight[i * config.cities_number + j]);
    context.total_weight[j] -= smooth_relu(context.weight[j * config.cities_number + i]);

    context.weight[i * config.cities_number + j] += weight_delta;
    context.weight[j * config.cities_number + i] += weight_delta;

    context.total_weight[i] += smooth_relu(context.weight[i * config.cities_number + j]);
    context.total_weight[j] += smooth_relu(context.weight[j * config.cities_number + i]);
}


void identify_candidates_for_each_node(const Config& config, Context& context, const double* metric, bool is_reversed) {
	for (int i = 0; i < config.cities_number; ++i) {
        std::iota(context.buffer.begin(), context.buffer.end(), 0);  // just a simple range(0, n), vector should be filled to use std::iota

        std::nth_element(context.buffer.begin(), context.buffer.begin() + config.candidates_number, context.buffer.end(), [&](int u, int v) {
            if (i == u) { return false; }
            if (i == v) { return true; }
            return static_cast<bool>((metric[i * config.cities_number + u] < metric[i * config.cities_number + v]) ^ is_reversed);
        });
		
		for (int j = 0; j < config.candidates_number; ++j) {
			context.candidates[i * config.candidates_number + j] = context.buffer[j];
	    }
	}
}


int get_random_int_by_module(int mod) {
	return rand() % mod;
}


bool is_cities_same_or_adjacent(const Config& config, Context& context, int i, int j) {
    return (i == j || context.path[i].next == j || context.path[j].next == i);
}


void reverse_sub_path(Context& context, int i, int j) {
    int current_city = i;

    while (true) {
        std::swap(context.path[current_city].prev, context.path[current_city].next);

        if (current_city == j) { return; }

        current_city = context.path[current_city].prev;
    }
}


double expected_optimal_tsp_length_2d(long long n, double width, double height) {
    if (n <= 1 || width <= 0.0 || height <= 0.0) {
        return 0.0;
    }

    double area = width * height;
    double expected_length = BHH_CONSTANT_2D * std::sqrt(static_cast<double>(n) * area);
    return expected_length;
}