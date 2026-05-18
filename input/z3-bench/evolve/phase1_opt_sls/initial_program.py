"""
Phase 1: tune opt.* + sls.*.

Other namespaces (sat.*, smt.*, parallel.*) stay at baseline. Z3 4.13.x keys.
EVOLVE-BLOCK below is the only thing the LLM should change. Keys may be added,
removed, or have values modified.

Do NOT modify locked keys (sls.random_seed; sat./smt./parallel.* live in BASELINE
and stay there). Invalid Z3 keys cause evaluator to return 0.
"""
import pathlib
import sys

_SHARED = pathlib.Path(__file__).resolve().parent.parent / "shared"
sys.path.insert(0, str(_SHARED))

from baseline_params import BASELINE  # noqa: E402


# EVOLVE-BLOCK-START
OPT_SLS_OVERRIDES = {
    # opt.* — MaxSMT engine, MaxRes/RC2 knobs, optsmt engine
    "opt.priority": "pareto",                   # lex | pareto | box
    "opt.maxsat_engine": "wmax",                # maxres | pd-maxres | wmax | sortmax | rc2 | maxres-bin
    "opt.optsmt_engine": "basic",               # basic | farkas | symba
    "opt.enable_sat": True,
    "opt.enable_sls": True,
    "opt.enable_core_rotate": True,
    "opt.enable_lns": False,
    "opt.lns.threshold": 4,
    "opt.maxres.hill_climb": True,
    "opt.maxres.add_upper_bound_block": False,
    "opt.maxres.max_core_size": 3,
    "opt.maxres.max_correction_set_size": 3,
    "opt.maxres.maximize_assignment": False,
    "opt.maxres.pivot_on_correction_set": True,
    "opt.maxres.wmax": False,
    "opt.maxlex.enable": True,
    "opt.rc2.totalizer": True,
    "opt.pb.compile_equality": False,
    "opt.elim_01": True,

    # sls.* — stochastic local search (engaged via opt.enable_sls)
    "sls.early_prune": True,
    "sls.walksat": True,
    "sls.walksat_repick": True,
    "sls.walksat_ucb": True,
    "sls.walksat_ucb_constant": 20.0,
    "sls.walksat_ucb_forget": 0.1,
    "sls.walksat_ucb_init": False,
    "sls.walksat_ucb_noise": 0.0002,
    "sls.wp": 20,                               # walk probability (percent 0..100)
    "sls.parallel": False,
    "sls.random_offset": True,
    "sls.rescore": True,
    "sls.restart_base": 100,
    "sls.restart_init": False,
    "sls.track_unsat": False,
}
# EVOLVE-BLOCK-END


def get_params():
    p = dict(BASELINE)
    p.update(OPT_SLS_OVERRIDES)
    return p


def get_phase_overrides():
    """Used by extract_best.py — returns ONLY this phase's evolved dict."""
    return dict(OPT_SLS_OVERRIDES)
