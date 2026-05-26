# CpSolverParameters Reference (ortools 9.14.6206)

Total fields: **275**. Auto-categorized by name pattern; some fields
appear under a best-fit bucket. Always verify with the upstream proto:

`ortools/sat/sat_parameters.proto`. Setting any field on `cp_model.CpSolver().parameters`
via `setattr(s.parameters, name, value)` works for all of these.


## parallel/workers (19)

| name | type | default | enum / notes |
|---|---|---|---|
| `diversify_lns_params` | bool | False |  |
| `extra_subsolvers` | list<string> |  |  |
| `filter_subsolvers` | list<string> |  |  |
| `ignore_subsolvers` | list<string> |  |  |
| `interleave_search` | bool | False |  |
| `lb_relax_num_workers_threshold` | int32 | 16 |  |
| `log_subsolver_statistics` | bool | False |  |
| `num_full_subsolvers` | int32 | 0 |  |
| `num_search_workers` | int32 | 0 |  |
| `num_workers` | int32 | 0 |  |
| `share_binary_clauses` | bool | True |  |
| `share_glue_clauses` | bool | False |  |
| `share_glue_clauses_dtime` | double | 1.0 |  |
| `share_level_zero_bounds` | bool | True |  |
| `share_objective_bounds` | bool | True |  |
| `shared_tree_num_workers` | int32 | 0 |  |
| `subsolver_params` | list<msg:SatParameters> |  |  |
| `subsolvers` | list<string> |  |  |
| `use_objective_lb_search` | bool | False |  |

## seed/reproducibility (18)

| name | type | default | enum / notes |
|---|---|---|---|
| `ignore_names` | bool | True |  |
| `instantiate_all_variables` | bool | True |  |
| `lns_initial_deterministic_limit` | double | 0.1 |  |
| `log_prefix` | string | '' |  |
| `log_search_progress` | bool | False |  |
| `log_to_response` | bool | False |  |
| `log_to_stdout` | bool | True |  |
| `max_deterministic_time` | double | inf |  |
| `max_num_deterministic_batches` | int32 | 0 |  |
| `name` | string | '' |  |
| `permute_presolve_constraint_order` | bool | False |  |
| `permute_variable_randomly` | bool | False |  |
| `presolve_probing_deterministic_time_limit` | double | 30.0 |  |
| `probing_deterministic_time_limit` | double | 1.0 |  |
| `random_seed` | int32 | 1 |  |
| `shaving_deterministic_time_in_probing_search` | double | 0.001 |  |
| `shaving_search_deterministic_time` | double | 0.1 |  |
| `symmetry_detection_deterministic_time_limit` | double | 1.0 |  |

## timeout/limits (7)

| name | type | default | enum / notes |
|---|---|---|---|
| `max_memory_in_mb` | int64 | 10000 |  |
| `max_number_of_conflicts` | int64 | 9223372036854775807 |  |
| `max_presolve_iterations` | int32 | 3 |  |
| `max_time_in_seconds` | double | inf |  |
| `stop_after_first_solution` | bool | False |  |
| `stop_after_presolve` | bool | False |  |
| `stop_after_root_propagation` | bool | False |  |

## search/branching (22)

| name | type | default | enum / notes |
|---|---|---|---|
| `boolean_encoding_level` | int32 | 1 |  |
| `fix_variables_to_their_hinted_value` | bool | False |  |
| `hint_conflict_limit` | int32 | 10 |  |
| `initial_polarity` | enum | 1 | {POLARITY_TRUE, POLARITY_FALSE, POLARITY_RANDOM} |
| `initial_variables_activity` | double | 0.0 |  |
| `polarity_exploit_ls_hints` | bool | False |  |
| `polarity_rephase_increment` | int32 | 1000 |  |
| `preferred_variable_order` | enum | 0 | {IN_ORDER, IN_REVERSE_ORDER, IN_RANDOM_ORDER} |
| `random_polarity_ratio` | double | 0.0 |  |
| `repair_hint` | bool | False |  |
| `search_branching` | enum | 0 | {AUTOMATIC_SEARCH, FIXED_SEARCH, PORTFOLIO_SEARCH, LP_SEARCH, PSEUDO_COST_SEARCH, PORTFOLIO_WITH_QUICK_RESTART_SEARCH, HINT_SEARCH, PARTIAL_FIXED_SEARCH, RANDOMIZED_SEARCH} |
| `use_combined_no_overlap` | bool | False |  |
| `use_disjunctive_constraint_in_cumulative` | bool | True |  |
| `use_dual_scheduling_heuristics` | bool | True |  |
| `use_erwa_heuristic` | bool | False |  |
| `use_extended_probing` | bool | True |  |
| `use_implied_bounds` | bool | True |  |
| `use_lns` | bool | True |  |
| `use_lns_only` | bool | False |  |
| `use_objective_shaving_search` | bool | False |  |
| `use_phase_saving` | bool | True |  |
| `use_strong_propagation_in_disjunctive` | bool | False |  |

## conflict/restart (18)

| name | type | default | enum / notes |
|---|---|---|---|
| `binary_minimization_algorithm` | enum | 1 | {NO_BINARY_MINIMIZATION, BINARY_MINIMIZATION_FIRST, BINARY_MINIMIZATION_FIRST_WITH_TRANSITIVE_REDUCTION, BINARY_MINIMIZATION_WITH_REACHABILITY, EXPERIMENTAL_BINARY_MINIMIZATION} |
| `blocking_restart_multiplier` | double | 1.4 |  |
| `blocking_restart_window_size` | int32 | 5000 |  |
| `default_restart_algorithms` | string | 'LUBY_RESTART,LBD_MOVING_AVERAGE_RESTART,DL_MOVING_AVERAGE_RESTART' |  |
| `feasibility_jump_restart_factor` | int32 | 1 |  |
| `glucose_decay_increment` | double | 0.01 |  |
| `glucose_decay_increment_period` | int32 | 5000 |  |
| `glucose_max_decay` | double | 0.95 |  |
| `max_variable_activity_value` | double | 1e+100 |  |
| `minimization_algorithm` | enum | 2 | {NONE, SIMPLE, RECURSIVE, EXPERIMENTAL} |
| `restart_algorithms` | list<enum> |  | {NO_RESTART, LUBY_RESTART, DL_MOVING_AVERAGE_RESTART, LBD_MOVING_AVERAGE_RESTART, FIXED_RESTART} |
| `restart_dl_average_ratio` | double | 1.0 |  |
| `restart_lbd_average_ratio` | double | 1.0 |  |
| `restart_period` | int32 | 50 |  |
| `restart_running_window_size` | int32 | 50 |  |
| `subsumption_during_conflict_analysis` | bool | True |  |
| `use_blocking_restart` | bool | False |  |
| `variable_activity_decay` | double | 0.8 |  |

## clause db (8)

| name | type | default | enum / notes |
|---|---|---|---|
| `clause_cleanup_lbd_bound` | int32 | 5 |  |
| `clause_cleanup_ordering` | enum | 0 | {CLAUSE_ACTIVITY, CLAUSE_LBD} |
| `clause_cleanup_period` | int32 | 10000 |  |
| `clause_cleanup_protection` | enum | 0 | {PROTECTION_NONE, PROTECTION_ALWAYS, PROTECTION_LBD} |
| `clause_cleanup_ratio` | double | 0.5 |  |
| `clause_cleanup_target` | int32 | 0 |  |
| `pb_cleanup_increment` | int32 | 200 |  |
| `pb_cleanup_ratio` | double | 0.5 |  |

## presolve (27)

| name | type | default | enum / notes |
|---|---|---|---|
| `convert_intervals` | bool | True |  |
| `cp_model_presolve` | bool | True |  |
| `cp_model_probing_level` | int32 | 2 |  |
| `debug_crash_if_presolve_breaks_hint` | bool | False |  |
| `debug_max_num_presolve_operations` | int32 | 0 |  |
| `disable_constraint_expansion` | bool | False |  |
| `encode_complex_linear_constraint_with_integer` | bool | False |  |
| `expand_alldiff_constraints` | bool | False |  |
| `expand_reservoir_constraints` | bool | True |  |
| `expand_reservoir_using_circuit` | bool | False |  |
| `find_big_linear_overlap` | bool | True |  |
| `infer_all_diffs` | bool | True |  |
| `max_pairs_pairwise_reasoning_in_no_overlap_2d` | int32 | 1250 |  |
| `max_size_to_create_precedence_literals_in_disjunctive` | int32 | 60 |  |
| `merge_at_most_one_work_limit` | double | 100000000.0 |  |
| `merge_no_overlap_work_limit` | double | 1000000000000.0 |  |
| `mip_presolve_level` | int32 | 2 |  |
| `presolve_blocked_clause` | bool | True |  |
| `presolve_bva_threshold` | int32 | 1 |  |
| `presolve_bve_clause_weight` | int32 | 3 |  |
| `presolve_bve_threshold` | int32 | 500 |  |
| `presolve_extract_integer_enforcement` | bool | False |  |
| `presolve_inclusion_work_limit` | int64 | 100000000 |  |
| `presolve_substitution_level` | int32 | 1 |  |
| `presolve_use_bva` | bool | True |  |
| `remove_fixed_variables_early` | bool | True |  |
| `symmetry_level` | int32 | 2 |  |

## LP/cuts (31)

| name | type | default | enum / notes |
|---|---|---|---|
| `add_cg_cuts` | bool | True |  |
| `add_clique_cuts` | bool | True |  |
| `add_lin_max_cuts` | bool | True |  |
| `add_lp_constraints_lazily` | bool | True |  |
| `add_mir_cuts` | bool | True |  |
| `add_objective_cut` | bool | False |  |
| `add_rlt_cuts` | bool | True |  |
| `add_zero_half_cuts` | bool | True |  |
| `cut_active_count_decay` | double | 0.8 |  |
| `cut_cleanup_target` | int32 | 1000 |  |
| `cut_level` | int32 | 1 |  |
| `cut_max_active_count_value` | double | 10000000000.0 |  |
| `exploit_all_lp_solution` | bool | True |  |
| `exploit_all_precedences` | bool | False |  |
| `exploit_integer_lp_solution` | bool | True |  |
| `feasibility_jump_linearization_level` | int32 | 2 |  |
| `linearization_level` | int32 | 1 |  |
| `lp_dual_tolerance` | double | 1e-07 |  |
| `lp_primal_tolerance` | double | 1e-07 |  |
| `max_all_diff_cut_size` | int32 | 64 |  |
| `max_consecutive_inactive_count` | int32 | 100 |  |
| `max_cut_rounds_at_level_zero` | int32 | 1 |  |
| `max_integer_rounding_scaling` | int32 | 600 |  |
| `max_num_cuts` | int32 | 10000 |  |
| `new_constraints_batch_size` | int32 | 50 |  |
| `only_add_cuts_at_level_zero` | bool | False |  |
| `use_energetic_reasoning_in_no_overlap_2d` | bool | False |  |
| `use_optimization_hints` | bool | True |  |
| `use_overload_checker_in_cumulative` | bool | False |  |
| `use_pb_resolution` | bool | False |  |
| `use_timetabling_in_no_overlap_2d` | bool | False |  |

## MIP bridge (11)

| name | type | default | enum / notes |
|---|---|---|---|
| `mip_automatically_scale_variables` | bool | True |  |
| `mip_check_precision` | double | 0.0001 |  |
| `mip_compute_true_objective_bound` | bool | True |  |
| `mip_drop_tolerance` | double | 1e-16 |  |
| `mip_max_activity_exponent` | int32 | 53 |  |
| `mip_max_bound` | double | 10000000.0 |  |
| `mip_max_valid_magnitude` | double | 1e+20 |  |
| `mip_scale_large_domain` | bool | False |  |
| `mip_treat_high_magnitude_bounds_as_infinity` | bool | False |  |
| `mip_var_scaling` | double | 1.0 |  |
| `mip_wanted_precision` | double | 1e-06 |  |

## LNS (1)

| name | type | default | enum / notes |
|---|---|---|---|
| `lns_initial_difficulty` | double | 0.5 |  |

## feasibility jump/pump (10)

| name | type | default | enum / notes |
|---|---|---|---|
| `feasibility_jump_batch_dtime` | double | 0.1 |  |
| `feasibility_jump_decay` | double | 0.95 |  |
| `feasibility_jump_enable_restarts` | bool | True |  |
| `feasibility_jump_max_expanded_constraint_size` | int32 | 500 |  |
| `feasibility_jump_var_perburbation_range_ratio` | double | 0.2 |  |
| `feasibility_jump_var_randomization_probability` | double | 0.05 |  |
| `use_feasibility_jump` | bool | True |  |
| `use_feasibility_pump` | bool | True |  |
| `violation_ls_compound_move_probability` | double | 0.5 |  |
| `violation_ls_perturbation_period` | int32 | 100 |  |

## optimization (10)

| name | type | default | enum / notes |
|---|---|---|---|
| `auto_detect_greater_than_at_least_one_of` | bool | True |  |
| `binary_search_num_conflicts` | int32 | -1 |  |
| `exploit_best_solution` | bool | False |  |
| `exploit_relaxation_solution` | bool | False |  |
| `minimize_reduction_during_pb_resolution` | bool | False |  |
| `optimize_with_core` | bool | False |  |
| `optimize_with_lb_tree_search` | bool | False |  |
| `optimize_with_max_hs` | bool | False |  |
| `use_absl_random` | bool | False |  |
| `use_precedences_in_disjunctive_constraint` | bool | True |  |

## debug/output (3)

| name | type | default | enum / notes |
|---|---|---|---|
| `debug_crash_on_bad_hint` | bool | False |  |
| `debug_postsolve_with_full_solver` | bool | False |  |
| `fp_rounding` | enum | 2 | {NEAREST_INTEGER, LOCK_BASED, ACTIVE_LOCK_BASED, PROPAGATION_ASSISTED} |

## other/uncategorized (90)

| name | type | default | enum / notes |
|---|---|---|---|
| `absolute_gap_limit` | double | 0.0001 |  |
| `also_bump_variables_in_conflict_reasons` | bool | False |  |
| `at_most_one_max_expansion_size` | int32 | 3 |  |
| `catch_sigint_signal` | bool | True |  |
| `clause_activity_decay` | double | 0.999 |  |
| `core_minimization_level` | int32 | 2 |  |
| `count_assumption_levels_in_lbd` | bool | True |  |
| `cover_optimization` | bool | True |  |
| `cp_model_use_sat_presolve` | bool | True |  |
| `detect_linearized_product` | bool | False |  |
| `detect_table_with_cost` | bool | False |  |
| `encode_cumulative_as_reservoir` | bool | False |  |
| `enumerate_all_solutions` | bool | False |  |
| `exploit_objective` | bool | True |  |
| `fill_additional_solutions_in_response` | bool | False |  |
| `fill_tightened_domains_in_response` | bool | False |  |
| `filter_sat_postsolve_clauses` | bool | False |  |
| `find_multiple_cores` | bool | True |  |
| `inprocessing_dtime_ratio` | double | 0.2 |  |
| `inprocessing_minimization_dtime` | double | 1.0 |  |
| `inprocessing_minimization_use_all_orderings` | bool | False |  |
| `inprocessing_minimization_use_conflict_analysis` | bool | True |  |
| `inprocessing_probing_dtime` | double | 1.0 |  |
| `interleave_batch_size` | int32 | 0 |  |
| `keep_all_feasible_solutions_in_presolve` | bool | False |  |
| `keep_symmetry_in_presolve` | bool | False |  |
| `linear_split_size` | int32 | 100 |  |
| `max_alldiff_domain_size` | int32 | 256 |  |
| `max_clause_activity_value` | double | 1e+20 |  |
| `max_domain_size_when_encoding_eq_neq_constraints` | int32 | 16 |  |
| `max_lin_max_size_for_expansion` | int32 | 0 |  |
| `max_num_intervals_for_timetable_edge_finding` | int32 | 100 |  |
| `max_sat_assumption_order` | enum | 0 | {DEFAULT_ASSUMPTION_ORDER, ORDER_ASSUMPTION_BY_DEPTH, ORDER_ASSUMPTION_BY_WEIGHT} |
| `max_sat_reverse_assumption_order` | bool | False |  |
| `max_sat_stratification` | enum | 1 | {STRATIFICATION_NONE, STRATIFICATION_DESCENT, STRATIFICATION_ASCENT} |
| `maximum_regions_to_split_in_disconnected_no_overlap_2d` | int32 | 0 |  |
| `min_orthogonality_for_lp_constraints` | double | 0.05 |  |
| `minimize_shared_clauses` | bool | True |  |
| `new_linear_propagation` | bool | True |  |
| `no_overlap_2d_boolean_relations_limit` | int32 | 10 |  |
| `num_conflicts_before_strategy_changes` | int32 | 0 |  |
| `num_violation_ls` | int32 | 0 |  |
| `only_solve_ip` | bool | False |  |
| `polish_lp_solution` | bool | False |  |
| `probing_num_combinations_limit` | int32 | 20000 |  |
| `propagation_loop_detection_factor` | double | 10.0 |  |
| `pseudo_cost_reliability_threshold` | int64 | 100 |  |
| `push_all_tasks_toward_start` | bool | False |  |
| `random_branches_ratio` | double | 0.0 |  |
| `randomize_search` | bool | False |  |
| `relative_gap_limit` | double | 0.0 |  |
| `root_lp_iterations` | int32 | 2000 |  |
| `routing_cut_dp_effort` | double | 10000000.0 |  |
| `routing_cut_max_infeasible_path_length` | int32 | 6 |  |
| `routing_cut_subset_size_for_binary_relation_bound` | int32 | 0 |  |
| `routing_cut_subset_size_for_exact_binary_relation_bound` | int32 | 8 |  |
| `routing_cut_subset_size_for_shortest_paths_bound` | int32 | 8 |  |
| `routing_cut_subset_size_for_tight_binary_relation_bound` | int32 | 0 |  |
| `save_lp_basis_in_lb_tree_search` | bool | False |  |
| `search_random_variable_pool_size` | int64 | 0 |  |
| `shared_tree_balance_tolerance` | int32 | 1 |  |
| `shared_tree_max_nodes_per_worker` | int32 | 10000 |  |
| `shared_tree_open_leaves_per_worker` | double | 2.0 |  |
| `shared_tree_split_strategy` | enum | 0 | {SPLIT_STRATEGY_AUTO, SPLIT_STRATEGY_DISCREPANCY, SPLIT_STRATEGY_OBJECTIVE_LB, SPLIT_STRATEGY_BALANCED_TREE, SPLIT_STRATEGY_FIRST_PROPOSAL} |
| `shared_tree_worker_enable_phase_sharing` | bool | True |  |
| `shared_tree_worker_enable_trail_sharing` | bool | True |  |
| `shared_tree_worker_min_restarts_per_subtree` | int32 | 1 |  |
| `shaving_search_threshold` | int64 | 64 |  |
| `solution_pool_size` | int32 | 3 |  |
| `strategy_change_increase_ratio` | double | 0.0 |  |
| `table_compression_level` | int32 | 2 |  |
| `use_all_different_for_circuit` | bool | False |  |
| `use_area_energetic_reasoning_in_no_overlap_2d` | bool | False |  |
| `use_conservative_scale_overload_checker` | bool | False |  |
| `use_dynamic_precedence_in_cumulative` | bool | False |  |
| `use_dynamic_precedence_in_disjunctive` | bool | False |  |
| `use_exact_lp_reason` | bool | True |  |
| `use_hard_precedences_in_cumulative` | bool | False |  |
| `use_lb_relax_lns` | bool | True |  |
| `use_linear3_for_no_overlap_2d_precedences` | bool | True |  |
| `use_ls_only` | bool | False |  |
| `use_optional_variables` | bool | False |  |
| `use_probing_search` | bool | False |  |
| `use_rins_lns` | bool | True |  |
| `use_sat_inprocessing` | bool | True |  |
| `use_shared_tree_search` | bool | False |  |
| `use_symmetry_in_lp` | bool | False |  |
| `use_timetable_edge_finding_in_cumulative` | bool | False |  |
| `use_try_edge_reasoning_in_no_overlap_2d` | bool | False |  |
| `variables_shaving_level` | int32 | -1 |  |
