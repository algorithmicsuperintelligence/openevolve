"""
Baseline Z3 parameters (applied_params_hash 543b29...) from problems.jsonl.
DO NOT MODIFY. Imported by all phases.
"""

BASELINE = {
    "opt.enable_core_rotate": True,
    "opt.enable_sat": True,
    "opt.enable_sls": True,
    "opt.maxres.hill_climb": True,
    "opt.maxsat_engine": "wmax",
    "opt.priority": "pareto",
    "opt.rc2.totalizer": True,
    "parallel.enable": False,
    "sat.branching.heuristic": "vsids",
    "sat.pb.solver": "totalizer",
    "sat.phase": "caching",
    "sat.random_seed": 0,
    "sat.restart": "geometric",
    "sat.threads": 1,
    "sls.random_seed": 0,
    "smt.phase_selection": 3,
    "smt.random_seed": 0,
    "smt.threads": 1,
    "threads": 1,
}

LOCKED = {
    "sat.random_seed": 0,
    "smt.random_seed": 0,
    "sls.random_seed": 0,
    "parallel.enable": False,
}
