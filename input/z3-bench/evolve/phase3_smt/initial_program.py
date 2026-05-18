"""
Phase 3: tune smt.* (SMT core — theories, quantifier instantiation, arith).

Loads phase1_best.json (opt./sls.*) and phase2_best.json (sat.*) as locked.
parallel.* stays at baseline. EVOLVE-BLOCK is SMT_OVERRIDES below.

NOTE: smt.auto_config=True (default) can silently override other smt.* options;
we force False here so the LLM's choices stick.

Do NOT modify smt.random_seed (locked). Invalid keys -> evaluator returns 0.
"""
import json
import pathlib
import sys

_SHARED = pathlib.Path(__file__).resolve().parent.parent / "shared"
sys.path.insert(0, str(_SHARED))

from baseline_params import BASELINE  # noqa: E402

_PHASE1 = (
    json.loads((_SHARED / "phase1_best.json").read_text())
    if (_SHARED / "phase1_best.json").exists()
    else {}
)
_PHASE2 = (
    json.loads((_SHARED / "phase2_best.json").read_text())
    if (_SHARED / "phase2_best.json").exists()
    else {}
)


# EVOLVE-BLOCK-START
SMT_OVERRIDES = {
    # Core SMT control
    "smt.auto_config": False,                  # FORCED False so other smt.* take effect
    "smt.logic": "",                           # "" | "QF_LIA" | "QF_LRA" | "LIA" | ...
    "smt.threads": 1,                          # keep 1; parallel.enable is locked false
    "smt.threads.cube_frequency": 2,
    "smt.threads.max_conflicts": 400,
    "smt.cube_depth": 1,
    "smt.relevancy": 2,                        # 0 | 1 | 2
    "smt.case_split": 1,                       # 0=activity | 1=random | 2=theory | 3=relevancy | 5 | 6
    "smt.phase_selection": 3,                  # 0=always_false | 1=always_true | 2=basic_caching | 3=caching | 4=random | 5=occurrence
    "smt.phase_caching_on": 400,
    "smt.phase_caching_off": 100,
    "smt.restart_strategy": 1,                 # 0=geometric | 1=inner_outer | 2=luby | 3=fixed | 4=arithmetic
    "smt.restart.factor": 1.1,
    "smt.lemma_gc_strategy": 0,                # 0=fixed | 1=geometric | 2=at_restart | 3=none

    # Lemma / unit delay
    "smt.delay_units": False,
    "smt.delay_units_threshold": 32,

    # Dynamic ackermann (dack)
    "smt.dack": 1,                             # 0=off | 1=on
    "smt.dack.eq": False,
    "smt.dack.factor": 0.1,
    "smt.dack.gc": 2000,
    "smt.dack.gc_inv_decay": 0.8,
    "smt.dack.threshold": 10,

    # Generic
    "smt.elim_unconstrained": True,
    "smt.ematching": True,
    "smt.macro_finder": False,
    "smt.quasi_macros": False,
    "smt.propagate_values": True,
    "smt.pull_nested_quantifiers": False,
    "smt.refine_inj_axioms": True,
    "smt.solve_eqs": True,
    "smt.solve_eqs_max_occs": 4294967295,
    "smt.theory_aware_branching": False,
    "smt.theory_case_split": False,
    "smt.dt_lazy_splits": 1,
    "smt.induction": False,

    # Core extension / minimization
    "smt.core.extend_patterns": False,
    "smt.core.extend_nonlocal_patterns": False,
    "smt.core.extend_patterns.max_distance": 4294967295,
    "smt.core.minimize": False,
    "smt.core.validate": False,

    # MBQI / quantifier instantiation
    "smt.mbqi": True,
    "smt.mbqi.max_iterations": 1000,
    "smt.mbqi.max_cexs": 1,
    "smt.mbqi.max_cexs_incr": 0,
    "smt.mbqi.force_template": 10,
    "smt.qi.eager_threshold": 10.0,
    "smt.qi.lazy_threshold": 20.0,
    "smt.qi.max_instances": 4294967295,
    "smt.qi.max_multi_patterns": 0,
    "smt.qi.cost": "(+ weight generation)",
    "smt.qi.quick_checker": 0,                 # 0=no | 1=unsat | 2=both

    # Arithmetic theory (workload has 13k Int + 40 Real — IMPORTANT)
    "smt.arith.solver": 6,                     # 2=simplex | 5=infinitary_lra | 6=lra
    "smt.arith.simplex_strategy": 0,
    "smt.arith.propagation_mode": 1,           # 0=none | 1=at_pivot | 2=cheap
    "smt.arith.propagate_eqs": True,
    "smt.arith.eager_eq_axioms": True,
    "smt.arith.branch_cut_ratio": 2,
    "smt.arith.bprop_on_pivoted_rows": True,
    "smt.arith.enable_hnf": True,
    "smt.arith.greatest_error_pivot": False,
    "smt.arith.ignore_int": False,
    "smt.arith.int_eq_branch": False,
    "smt.arith.min": False,
    "smt.arith.random_initial_value": False,
    "smt.arith.rep_freq": 0,
    "smt.arith.auto_config_simplex": False,

    # Nonlinear arith (mostly off for LIA-heavy workload; expose anyway)
    "smt.arith.nl": True,
    "smt.arith.nl.branching": True,
    "smt.arith.nl.delay": 500,
    "smt.arith.nl.expp": False,
    "smt.arith.nl.gr_q": 10,
    "smt.arith.nl.grobner": True,
    "smt.arith.nl.grobner_cnfl_to_report": 1,
    "smt.arith.nl.grobner_eqs_growth": 10,
    "smt.arith.nl.grobner_expr_degree_growth": 2,
    "smt.arith.nl.grobner_expr_size_growth": 2,
    "smt.arith.nl.grobner_frequency": 4,
    "smt.arith.nl.grobner_max_simplified": 10000,
    "smt.arith.nl.grobner_subs_fixed": 1,
    "smt.arith.nl.horner": True,
    "smt.arith.nl.horner_frequency": 4,
    "smt.arith.nl.horner_row_length_limit": 10,
    "smt.arith.nl.horner_subs_fixed": 2,
    "smt.arith.nl.nra": True,
    "smt.arith.nl.order": True,
    "smt.arith.nl.rounds": 1024,
    "smt.arith.nl.tangents": True,

    # BV (light usage in this workload)
    "smt.bv.delay": True,
    "smt.bv.eager": True,
    "smt.bv.enable_int2bv": True,
    "smt.bv.reflect": True,
    "smt.bv.size_reduce": False,
    "smt.bv.solver": 0,

    # Array / PB
    "smt.array.extensional": True,
    "smt.array.weak": False,
    "smt.pb.conflict_frequency": 1000,
    "smt.pb.learn_complements": True,
}
# EVOLVE-BLOCK-END


def get_params():
    p = dict(BASELINE)
    p.update(_PHASE1)
    p.update(_PHASE2)
    p.update(SMT_OVERRIDES)
    return p


def get_phase_overrides():
    return dict(SMT_OVERRIDES)
