"""
Runtime knob loader for z3-bench scripts.

Reads custom keys from ../config.yaml (top-level), with env var override.
openevolve's dacite parser silently ignores unknown top-level keys, so we
share the same file rather than introducing a second config.

Priority: env var > config.yaml > default.
"""
import os
import pathlib

_HERE = pathlib.Path(__file__).resolve().parent
_CONFIG_YAML = _HERE.parent / "config.yaml"   # input/z3-bench/evolve/config.yaml

_cache = None


def _load():
    global _cache
    if _cache is not None:
        return _cache
    if not _CONFIG_YAML.exists():
        _cache = {}
        return _cache
    try:
        import yaml
        _cache = yaml.safe_load(_CONFIG_YAML.read_text()) or {}
    except Exception:
        _cache = {}
    return _cache


def parallel_solvers(default=1):
    """
    Concurrent z3 worker subprocesses per stage.
    Env OPENEVOLVE_PARALLEL_SOLVERS > config.yaml parallel_solvers > default.
    """
    env = os.environ.get("OPENEVOLVE_PARALLEL_SOLVERS")
    if env is not None:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    val = _load().get("parallel_solvers", default)
    try:
        return max(1, int(val))
    except (ValueError, TypeError):
        return default


def cascade_threshold(index, default):
    """
    Read evaluator.cascade_thresholds[index] from config.yaml.
    Used by evaluator.evaluate_stage3 for the internal stage3→stage4 gate
    (openevolve's cascade hardcodes only 3 stage slots, so stage4 is chained
    inside evaluate_stage3 using thresholds[2]).
    """
    cfg = _load().get("evaluator") or {}
    thresholds = cfg.get("cascade_thresholds") or []
    if index < len(thresholds):
        try:
            return float(thresholds[index])
        except (ValueError, TypeError):
            pass
    return default
