"""
Shared runtime knob loader. Used by all bench evolve pipelines.

Reads custom keys from <bench>/evolve/config.yaml (top-level), with env var
override. openevolve's dacite parser silently ignores unknown top-level keys,
so each bench shares the same config file rather than introducing a second.

Priority: env var > config.yaml > default.

Callers pass the path to config.yaml. Per-bench `shared/runtime.py` wrappers
resolve their own config path and re-export the helpers below.
"""
import os
import pathlib

_cache = {}


def _load(config_path):
    p = pathlib.Path(config_path)
    key = str(p)
    if key in _cache:
        return _cache[key]
    if not p.exists():
        _cache[key] = {}
        return _cache[key]
    try:
        import yaml
        _cache[key] = yaml.safe_load(p.read_text()) or {}
    except Exception:
        _cache[key] = {}
    return _cache[key]


def parallel_solvers(config_path, default=1):
    """
    Concurrent solver worker subprocesses per stage.
    Env OPENEVOLVE_PARALLEL_SOLVERS > config.yaml parallel_solvers > default.
    """
    env = os.environ.get("OPENEVOLVE_PARALLEL_SOLVERS")
    if env is not None:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    val = _load(config_path).get("parallel_solvers", default)
    try:
        return max(1, int(val))
    except (ValueError, TypeError):
        return default


def core_range():
    """
    Parse OPENEVOLVE_CORE_RANGE=START-END (or single "N"). Returns list of
    core IDs to lease, or None if env unset.

    Caller decides what to do with None:
      - typical default = list(range(1, parallel_solvers() + 1))
        (cores 1..N, core 0 reserved for kernel housekeeping).

    Range size implicitly caps concurrency: n_parallel = len(core_range).
    """
    env = os.environ.get("OPENEVOLVE_CORE_RANGE")
    if not env:
        return None
    env = env.strip()
    try:
        if "-" in env:
            lo_s, hi_s = env.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                lo, hi = hi, lo
            return list(range(lo, hi + 1))
        return [int(env)]
    except ValueError:
        return None


def alloc_core_blocks(cores, workers_per_solve):
    """Floor-chunk a core list into blocks of `workers_per_solve` cores each.

    Used when a single solver subprocess must be pinned to W cores (e.g. CP-SAT
    with num_search_workers=W). Leftover cores at the tail are dropped — this
    keeps every concurrent solve identical in CPU budget so benchmark timings
    stay comparable.

    Examples:
        alloc_core_blocks([1,2,3,4,5,6], 1) -> [[1],[2],[3],[4],[5],[6]]
        alloc_core_blocks([1,2,3,4,5,6], 4) -> [[1,2,3,4]]            # 5,6 dropped
        alloc_core_blocks([1,2,3,4,5,6], 6) -> [[1,2,3,4,5,6]]
        alloc_core_blocks([1,2,3], 4)       -> []                     # not enough
    """
    cores = list(cores)
    try:
        w = max(1, int(workers_per_solve))
    except (TypeError, ValueError):
        w = 1
    n = len(cores) // w
    return [cores[i * w:(i + 1) * w] for i in range(n)]


def cascade_threshold(config_path, index, default):
    """
    Read evaluator.cascade_thresholds[index] from config.yaml.
    Used by evaluator.evaluate_stage3 for the internal stage3→stage4 gate
    (openevolve cascade hardcodes only 3 stage slots).
    """
    cfg = _load(config_path).get("evaluator") or {}
    thresholds = cfg.get("cascade_thresholds") or []
    if index < len(thresholds):
        try:
            return float(thresholds[index])
        except (ValueError, TypeError):
            pass
    return default
