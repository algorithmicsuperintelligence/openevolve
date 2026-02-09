#pragma once

#include <vector>
#include <algorithm>


// Enhanced candidate selection with spatial diversity
void identify_candidates_lk_style(const Config& config, Context& context, const double* metric, bool is_reversed) {
    for (int i = 0; i < config.cities_number; ++i) {
        std::vector<std::pair<double, int>> candidates;
        candidates.reserve(config.cities_number - 1);
        
        for (int j = 0; j < config.cities_number; ++j) {
            if (i == j) continue;
            double score = metric[i * config.cities_number + j];
            candidates.emplace_back(score, j);
        }
        
        // Sort by primary metric
        if (is_reversed) {
            std::partial_sort(candidates.begin(), candidates.begin() + config.candidates_number, candidates.end(),
                std::greater<std::pair<double, int>>());
        } else {
            std::partial_sort(candidates.begin(), candidates.begin() + config.candidates_number, candidates.end());
        }
        
        // Enhanced candidate diversity: mix nearest neighbors with spatially diverse candidates
        std::vector<int> final_candidates;
        final_candidates.reserve(config.candidates_number);
        
        // Take 70% nearest neighbors
        int nn_count = config.candidates_number * 0.7;
        for (int j = 0; j < nn_count && j < candidates.size(); ++j) {
            final_candidates.push_back(candidates[j].second);
        }
        
        // Add 30% diverse candidates from different distance ranges
        if (candidates.size() > nn_count) {
            int step = std::max(1, (int)(candidates.size() - nn_count) / (config.candidates_number - nn_count));
            for (int j = nn_count; j < candidates.size() && final_candidates.size() < config.candidates_number; j += step) {
                if (std::find(final_candidates.begin(), final_candidates.end(), candidates[j].second) == final_candidates.end()) {
                    final_candidates.push_back(candidates[j].second);
                }
            }
        }
        
        // Fill remaining slots with nearest neighbors if needed
        for (int j = 0; j < candidates.size() && final_candidates.size() < config.candidates_number; ++j) {
            if (std::find(final_candidates.begin(), final_candidates.end(), candidates[j].second) == final_candidates.end()) {
                final_candidates.push_back(candidates[j].second);
            }
        }
        
        // Copy to context
        for (int j = 0; j < config.candidates_number && j < final_candidates.size(); ++j) {
            context.candidates[i * config.candidates_number + j] = final_candidates[j];
        }
    }
}


// Enhanced 3-opt implementation testing all possible move types
bool apply_3_opt_move(const Config& config, Context& context, int i, int j, int k) {
    if (i == j || j == k || i == k) return false;
    
    int i_next = context.path[i].next;
    int j_next = context.path[j].next; 
    int k_next = context.path[k].next;

    if (i_next == j || j_next == k || k_next == i) return false;

    if (config.distance_type == DistanceType::Double) {
        double current = get_distance_double(config, context, i, i_next) +
                        get_distance_double(config, context, j, j_next) +
                        get_distance_double(config, context, k, k_next);
        
        double best_new_dist = current;
        int best_case = 0;
        
        // Test all 4 possible 3-opt moves
        double case1 = get_distance_double(config, context, i, j) +
                      get_distance_double(config, context, i_next, k) +
                      get_distance_double(config, context, j_next, k_next);
        
        double case2 = get_distance_double(config, context, i, j_next) +
                      get_distance_double(config, context, j, k) +
                      get_distance_double(config, context, i_next, k_next);
        
        double case3 = get_distance_double(config, context, i, k) +
                      get_distance_double(config, context, j_next, i_next) +
                      get_distance_double(config, context, j, k_next);
        
        double case4 = get_distance_double(config, context, i, j_next) +
                      get_distance_double(config, context, j, k_next) +
                      get_distance_double(config, context, k, i_next);
        
        if (case1 < best_new_dist) { best_new_dist = case1; best_case = 1; }
        if (case2 < best_new_dist) { best_new_dist = case2; best_case = 2; }
        if (case3 < best_new_dist) { best_new_dist = case3; best_case = 3; }
        if (case4 < best_new_dist) { best_new_dist = case4; best_case = 4; }
        
        if (best_new_dist < current) {
            double delta = current - best_new_dist;
            
            // Apply the best move
            switch (best_case) {
                case 1:
                    reverse_sub_path(context, i_next, j);
                    reverse_sub_path(context, j_next, k);
                    context.path[i].next = j;
                    context.path[i_next].next = k;
                    context.path[j_next].next = k_next;
                    context.path[j].prev = i;
                    context.path[k].prev = i_next;
                    context.path[k_next].prev = j_next;
                    break;
                case 2:
                    reverse_sub_path(context, i_next, j);
                    context.path[i].next = j_next;
                    context.path[j].next = k;
                    context.path[i_next].next = k_next;
                    context.path[j_next].prev = i;
                    context.path[k].prev = j;
                    context.path[k_next].prev = i_next;
                    break;
                case 3:
                    reverse_sub_path(context, i_next, j);
                    reverse_sub_path(context, j_next, k);
                    context.path[i].next = k;
                    context.path[j_next].next = i_next;
                    context.path[j].next = k_next;
                    context.path[k].prev = i;
                    context.path[i_next].prev = j_next;
                    context.path[k_next].prev = j;
                    break;
                case 4:
                    context.path[i].next = j_next;
                    context.path[j].next = k_next;
                    context.path[k].next = i_next;
                    context.path[j_next].prev = i;
                    context.path[k_next].prev = j;
                    context.path[i_next].prev = k;
                    break;
            }
            
            context.path_distance_double -= delta;
            return true;
        }
    }
    // Similar implementations for Int32 and Int64 would go here...
    
    return false;
}