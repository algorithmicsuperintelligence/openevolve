"""
Generate `<bench>/evolve/final_program.py` — self-contained canonical final
artifact from the last phase's best_program.py.

CLI: `python -m _lib.finalize <bench>`

Self-containment rules:
  - z3 phase4_unified style: `prepare_phase` already materialized the
    EVOLVE-BLOCK with the union of prior-phase winners as a literal dict
    (`UNIFIED_OVERRIDES = {...}`). Verbatim copy is already self-contained.

  - cpsat phase5 style: `_PHASE4 = _load_prev_dict("phase4_best.json")` reads
    `cache/phase4_best.json` at import time. Finalize replaces these calls
    with the literal dict from the JSON so the output file does NOT depend
    on `cache/` being present alongside.

Patterns recognized:
  `<NAME> = _load_prev_dict("<json>")`     → literal dict
  `<NAME> = _load_prev_buckets("<json>")`  → literal list[(upper, dict)]
                                              (None → float('inf'))
  `<NAME> = _load_prev("<json>")`          → literal dict (alias)

The helper function definitions remain in source (harmless — dead code).
`cache/` path resolution and JSON loading are removed by the replacement.

Source priority:
  1. `<last_phase>/openevolve_output/best/best_program.py`
  2. `<last_phase>/openevolve_output/checkpoints/checkpoint_*/best_program.py`
     (highest combined_score)
"""

import argparse
import json
import pathlib
import pprint
import re
import shutil
import sys

from _lib import bench_paths


_PAT_DICT = re.compile(
    r'^(\s*)(_[A-Za-z0-9_]+)\s*=\s*_load_prev(?:_dict)?\(\s*["\']([^"\']+)["\']\s*\)\s*$',
    re.M,
)
_PAT_BUCKETS = re.compile(
    r'^(\s*)(_[A-Za-z0-9_]+)\s*=\s*_load_prev_buckets\(\s*["\']([^"\']+)["\']\s*\)\s*$',
    re.M,
)


def _format_buckets(raw):
    items = []
    for entry in raw:
        if not (isinstance(entry, (list, tuple)) and len(entry) == 2):
            raise ValueError(f"bucket entry malformed: {entry!r}")
        upper, override = entry
        upper_src = "float('inf')" if upper is None else repr(upper)
        ov_src = pprint.pformat(override, width=100, sort_dicts=True)
        items.append(f"({upper_src}, {ov_src})")
    return "[" + ", ".join(items) + "]"


def _resolve_load_prev(src, cache_dir, replaced):
    """Substitute every `_load_prev{_dict,_buckets}("...json")` assignment
    with its resolved literal. Appends `(name, fname, kind)` tuples to
    `replaced` for the summary."""

    def repl_dict(m):
        indent, name, fname = m.group(1), m.group(2), m.group(3)
        path = cache_dir / fname
        if not path.exists():
            replaced.append((name, fname, "dict-missing"))
            return f"{indent}{name} = {{}}"
        data = json.loads(path.read_text())
        replaced.append((name, fname, "dict"))
        return f"{indent}{name} = " + pprint.pformat(data, width=100, sort_dicts=True)

    def repl_buckets(m):
        indent, name, fname = m.group(1), m.group(2), m.group(3)
        path = cache_dir / fname
        if not path.exists():
            replaced.append((name, fname, "buckets-missing"))
            return f"{indent}{name} = None"
        raw = json.loads(path.read_text())
        replaced.append((name, fname, "buckets"))
        return f"{indent}{name} = " + _format_buckets(raw)

    # Order matters — handle buckets before dict (dict regex would also match).
    src = _PAT_BUCKETS.sub(repl_buckets, src)
    src = _PAT_DICT.sub(repl_dict, src)
    return src


def _pick_from_checkpoints(phase_dir):
    ckpt_root = phase_dir / "openevolve_output" / "checkpoints"
    best_py, best_score, best_ck = None, float("-inf"), None
    for ck in sorted(ckpt_root.glob("checkpoint_*")):
        info = ck / "best_program_info.json"
        prog = ck / "best_program.py"
        if not info.exists() or not prog.exists():
            continue
        try:
            sc = float(
                json.loads(info.read_text()).get("metrics", {}).get("combined_score", float("-inf"))
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
        if sc > best_score:
            best_score, best_py, best_ck = sc, prog, ck
    return best_py, best_score, best_ck


def finalize(bench_root, from_checkpoints=False):
    bench_root = pathlib.Path(bench_root).resolve()
    cfg = bench_paths.load_config(bench_root)
    phases = (cfg.get("bench") or {}).get("phases") or []
    if not phases:
        raise SystemExit("bench.phases missing in config.yaml")

    last_dir = phases[-1]["dir"]
    last_phase = bench_root / last_dir
    cache_dir = bench_paths.cache_dir(bench_root)

    if from_checkpoints:
        best_py, score, ck = _pick_from_checkpoints(last_phase)
        if best_py is None:
            raise SystemExit(f"no checkpoint best_program.py under {last_phase}")
        print(f"[finalize] from-checkpoints: picked {ck.name} " f"(combined_score={score:.4f})")
    else:
        best_py = last_phase / "openevolve_output" / "best" / "best_program.py"
        if not best_py.exists():
            best_py_alt, score, ck = _pick_from_checkpoints(last_phase)
            if best_py_alt is None:
                raise SystemExit(
                    f"no best_program.py at {best_py} and no checkpoints under "
                    f"{last_phase / 'openevolve_output' / 'checkpoints'}"
                )
            best_py = best_py_alt
            print(
                f"[finalize] best/best_program.py missing — using checkpoint "
                f"{ck.name} (combined_score={score:.4f})"
            )

    src = best_py.read_text()
    header = (
        f"# AUTO-GENERATED by `python -m _lib.finalize {bench_root.parent.name}`.\n"
        f"# Source: {best_py.relative_to(bench_root)}\n"
        f"# All `_load_prev*()` calls below have been resolved against\n"
        f"# `cache/` and replaced with literal dicts so this file is self-\n"
        f"# contained and importable without `cache/` present.\n\n"
    )

    replaced = []
    resolved_src = _resolve_load_prev(src, cache_dir, replaced)
    if not resolved_src.startswith('"""') and not resolved_src.startswith("'''"):
        out_src = header + resolved_src
    else:
        # Insert header AFTER the module docstring.
        match = re.match(r'^(?P<q>"""|\'\'\')(.*?)(?P=q)\s*\n', resolved_src, re.S)
        if match:
            out_src = resolved_src[: match.end()] + "\n" + header + resolved_src[match.end() :]
        else:
            out_src = header + resolved_src

    suffix = bench_paths.variant_suffix(bench_root)
    final_name = f"final_program_{suffix}.py" if suffix else "final_program.py"
    out = bench_root / final_name
    out.write_text(out_src)
    rel_src = best_py.relative_to(bench_root)

    if replaced:
        print(f"[finalize] inlined {len(replaced)} _load_prev*() call(s):")
        for name, fname, kind in replaced:
            print(f"   {name} ← cache/{fname} ({kind})")
    else:
        print(
            "[finalize] no _load_prev*() calls to inline " "(EVOLVE-BLOCK already self-contained)"
        )
    print(f"[finalize] {rel_src} → {final_name} ({out.stat().st_size} bytes)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("bench", help="bench dir name (e.g. cpsat-bench)")
    ap.add_argument(
        "--from-checkpoints",
        action="store_true",
        help="scan checkpoint_*/ dirs and pick highest combined_score",
    )
    args = ap.parse_args(argv)
    finalize(bench_paths.resolve_bench(args.bench), from_checkpoints=args.from_checkpoints)


if __name__ == "__main__":
    main()
