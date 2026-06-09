"""
Shared path / config resolution used by every _lib module's CLI entry.

A "bench root" is the absolute path to `input/<bench>/evolve/`. Modules take
either the bench name (e.g. `"cpsat-bench"`) or an already-resolved bench
root path.
"""

import importlib.util
import os
import pathlib
import sys


def input_dir():
    """input/ — parent of this _lib package."""
    return pathlib.Path(__file__).resolve().parent.parent


def resolve_bench(bench_name_or_root):
    """Accept bench name (`cpsat-bench`) OR a full path to its evolve/."""
    p = pathlib.Path(bench_name_or_root)
    if p.is_absolute() and (p / "config.yaml").exists():
        return p.resolve()
    root = input_dir() / str(bench_name_or_root) / "evolve"
    if not root.exists():
        raise SystemExit(f"bench evolve dir not found: {root}")
    return root.resolve()


def solver_mode(bench_root):
    """`bench.solver_mode` from config.yaml. Selects the solver path / pipeline
    variant (e.g. z3-bench: "optimize" default vs "sat"). The value also drives
    cache + final_program suffixing so multiple modes coexist on disk."""
    m = (load_config(bench_root).get("bench") or {}).get("solver_mode")
    return m or "optimize"


def variant_suffix(bench_root):
    """Disk suffix for the active solver_mode. "" for the default ("optimize")
    so existing benches keep `cache/` + `final_program.py` unchanged; otherwise
    the mode name (e.g. "sat" → `cache-sat/`, `final_program_sat.py`)."""
    m = solver_mode(bench_root)
    return "" if m == "optimize" else m


def cache_dir(bench_root):
    base = pathlib.Path(bench_root).resolve() / "cache"
    s = variant_suffix(bench_root)
    return base.with_name(f"cache-{s}") if s else base


def raw_dir(bench_root):
    return pathlib.Path(bench_root).resolve().parent / "raw-data"


def problems_jsonl(bench_root):
    return pathlib.Path(bench_root).resolve().parent / "problems.jsonl"


def config_path(bench_root):
    return pathlib.Path(bench_root).resolve() / "config.yaml"


def params_json_path(bench_root):
    return pathlib.Path(bench_root).resolve() / "params.json"


def worker_path(bench_root):
    """Resolve `bench.worker_path` from config.yaml (relative to bench_root)."""
    bench_root = pathlib.Path(bench_root).resolve()
    cfg = load_config(bench_root)
    wp = (cfg.get("bench") or {}).get("worker_path") or "_solve_worker.py"
    p = bench_root / wp
    if not p.exists():
        raise SystemExit(f"worker_path not found: {p}")
    return p


def evaluation_cfg(bench_root):
    return (load_config(bench_root).get("bench") or {}).get("evaluation") or {}


def clustering_cfg(bench_root):
    return (load_config(bench_root).get("bench") or {}).get("clustering") or {}


_yaml_cache = {}


def load_config(bench_root):
    """Read `<bench>/evolve/config.yaml` (cached by path)."""
    path = config_path(bench_root)
    key = str(path)
    if key in _yaml_cache:
        return _yaml_cache[key]
    if not path.exists():
        _yaml_cache[key] = {}
        return _yaml_cache[key]
    import yaml

    _yaml_cache[key] = yaml.safe_load(path.read_text()) or {}
    return _yaml_cache[key]


def load_adapter(bench_root):
    """Import the bench's adapter.py and return the module."""
    bench_root = pathlib.Path(bench_root).resolve()
    adapter_path = bench_root / "adapter.py"
    if not adapter_path.exists():
        raise SystemExit(f"adapter.py not found: {adapter_path}")
    mod_name = f"_bench_adapter_{bench_root.parent.name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(mod_name, adapter_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bench_root_from_env():
    """Resolve bench root from OPENEVOLVE_BENCH_ROOT (set by run_phase.sh)
    when invoked as the openevolve evaluator. Returns None when unset so
    callers can decide on the fallback."""
    v = os.environ.get("OPENEVOLVE_BENCH_ROOT")
    if not v:
        return None
    return pathlib.Path(v).resolve()


def add_input_to_sys_path():
    """Ensure `input/` is on sys.path so `from _lib import ...` works from
    every entry point (CLI, openevolve eval subprocess, phase modules)."""
    p = str(input_dir())
    if p not in sys.path:
        sys.path.insert(0, p)
