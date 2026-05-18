"""
Phase 2: tune sat.* (CDCL SAT core).

Loads phase1_best.json (opt.*+sls.* winners) as locked. smt.* and parallel.*
stay at baseline. EVOLVE-BLOCK is SAT_OVERRIDES below.

Do NOT modify sat.random_seed (locked). Invalid keys -> evaluator returns 0.
"""
import json
import pathlib
import sys

_SHARED = pathlib.Path(__file__).resolve().parent.parent / "shared"
sys.path.insert(0, str(_SHARED))

from baseline_params import BASELINE  # noqa: E402

_PHASE1_FILE = _SHARED / "phase1_best.json"
_PHASE1 = (
    json.loads(_PHASE1_FILE.read_text()) if _PHASE1_FILE.exists() else {}
)


# EVOLVE-BLOCK-START
SAT_OVERRIDES = {
    # Branching / phase / restart
    "sat.branching.heuristic": "vsids",        # vsids | lrb | chb
    "sat.branching.anti_exploration": 0.4,
    "sat.phase": "caching",                    # always_false | always_true | basic_caching | random | caching
    "sat.phase.sticky": True,
    "sat.restart": "geometric",                # luby | geometric | ema | static
    "sat.restart.fast": True,
    "sat.restart.initial": 2,
    "sat.restart.factor": 1.5,
    "sat.restart.margin": 1.1,
    "sat.restart.emafastglue": 0.03,
    "sat.restart.emaslowglue": 1e-05,
    "sat.rephase.base": 1000,
    "sat.reorder.base": 4294967295,
    "sat.reorder.itau": 4.0,
    "sat.reorder.activity_scale": 100,
    "sat.random_freq": 0.01,
    "sat.variable_decay": 110,
    "sat.burst_search": 100,
    "sat.search.sat.conflicts": 400,
    "sat.search.unsat.conflicts": 400,
    "sat.backtrack.conflicts": 4000,
    "sat.backtrack.scopes": 100,

    # Garbage collection
    "sat.gc": "glue_psm",                      # glue | psm | glue_psm | dyn_psm
    "sat.gc.burst": False,
    "sat.gc.defrag": True,
    "sat.gc.increment": 500,
    "sat.gc.initial": 20000,
    "sat.gc.k": 7,
    "sat.gc.small_lbd": 3,
    "sat.minimize_lemmas": True,
    "sat.dyn.sub_res": True,

    # Preprocessing / simplification
    "sat.scc": True,
    "sat.scc.tr": True,
    "sat.elim_vars": True,
    "sat.elim_vars_bdd": True,
    "sat.elim_vars_bdd_delay": 3,
    "sat.subsumption": True,
    "sat.subsumption.limit": 100000000,
    "sat.asymm_branch": True,
    "sat.asymm_branch.all": False,
    "sat.asymm_branch.delay": 1,
    "sat.asymm_branch.limit": 100000000,
    "sat.asymm_branch.rounds": 2,
    "sat.asymm_branch.sampled": True,
    "sat.probing": True,
    "sat.probing_binary": True,
    "sat.probing_cache": True,
    "sat.probing_cache_limit": 1024,
    "sat.probing_limit": 5000000,
    "sat.propagate.prefetch": True,
    "sat.ate": True,
    "sat.acce": False,
    "sat.bce": False,
    "sat.bce_at": 2,
    "sat.bce_delay": 2,
    "sat.bca": False,
    "sat.binspr": False,
    "sat.cce": False,
    "sat.blocked_clause_limit": 100000000,
    "sat.retain_blocked_clauses": True,
    "sat.enable_pre_simplify": False,
    "sat.force_cleanup": False,
    "sat.inprocess.max": 4294967295,
    "sat.simplify.delay": 0,
    "sat.next_simplify1": 30000,

    # Cardinality / PB
    "sat.cardinality.solver": True,
    "sat.cardinality.encoding": "grouped",     # grouped | bimander | ordered | unate | circuit
    "sat.pb.solver": "totalizer",              # circuit | sorting | totalizer | solver | segmented | binary_merge
    "sat.pb.lemma_format": "cardinality",      # cardinality | pb
    "sat.pb.resolve": "cardinality",           # cardinality | rounding

    # Core minimization
    "sat.core.minimize": False,
    "sat.core.minimize_partial": False,

    # Threading (keep 1 for fair compare; parallel.enable stays locked false)
    "sat.threads": 1,

    # SLS-within-SAT (separate from opt-level sls.*)
    "sat.local_search": False,
    "sat.local_search_mode": "wsat",           # wsat | gsat
    "sat.local_search_threads": 0,
    "sat.ddfw_search": False,
    "sat.ddfw.threads": 0,
    "sat.ddfw.init_clause_weight": 8,
    "sat.ddfw.reinit_base": 10000,
    "sat.ddfw.restart_base": 100000,
    "sat.ddfw.use_reward_pct": 15,
    "sat.prob_search": False,

    # Cut/AIG/ANF preprocessing (default off for this workload)
    "sat.cut": False,
    "sat.cut.aig": False,
    "sat.cut.delay": 2,
    "sat.cut.dont_cares": True,
    "sat.cut.force": False,
    "sat.cut.lut": False,
    "sat.cut.npn3": False,
    "sat.cut.redundancies": True,
    "sat.cut.xor": False,
    "sat.anf": False,
    "sat.anf.delay": 2,
    "sat.anf.exlin": False,

    # Lookahead (mostly off; expose for solver-specific subproblems)
    "sat.lookahead.cube.cutoff": "depth",      # depth | freevars | psat | adaptive_freevars | adaptive_psat
    "sat.lookahead.cube.depth": 1,
    "sat.lookahead.cube.fraction": 0.4,
    "sat.lookahead.cube.freevars": 0.8,
    "sat.lookahead.cube.psat.clause_base": 2.0,
    "sat.lookahead.cube.psat.trigger": 5.0,
    "sat.lookahead.cube.psat.var_exp": 1.0,
    "sat.lookahead.delta_fraction": 1.0,
    "sat.lookahead.double": True,
    "sat.lookahead.global_autarky": False,
    "sat.lookahead.preselect": False,
    "sat.lookahead.reward": "march_cu",        # ternary | heule_schur | heule_unit | unit | march_cu
    "sat.lookahead.use_learned": False,
    "sat.lookahead_scores": False,
    "sat.lookahead_simplify": False,
    "sat.lookahead_simplify.bca": True,

    # Resolution-based simplification limits
    "sat.resolution.cls_cutoff1": 100000000,
    "sat.resolution.cls_cutoff2": 700000000,
    "sat.resolution.limit": 500000000,
    "sat.resolution.lit_cutoff_range1": 700,
    "sat.resolution.lit_cutoff_range2": 400,
    "sat.resolution.lit_cutoff_range3": 300,
    "sat.resolution.occ_cutoff": 10,
    "sat.resolution.occ_cutoff_range1": 8,
    "sat.resolution.occ_cutoff_range2": 5,
    "sat.resolution.occ_cutoff_range3": 3,
}
# EVOLVE-BLOCK-END


def get_params():
    p = dict(BASELINE)
    p.update(_PHASE1)
    p.update(SAT_OVERRIDES)
    return p


def get_phase_overrides():
    return dict(SAT_OVERRIDES)
