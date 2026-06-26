"""Triton-native ncu profiling: harness + collection + parsing, re-implemented
from the ncu-report-skill workflow for a JIT-compiled, multi-kernel-per-call
Triton program instead of an nvcc-compiled CUDA binary.

Differences from the CUDA workflow this is modeled on:
  - "Harness" is a throwaway Python script, not a compiled binary. There is no
    -lineinfo flag to pass; Triton embeds its own source debug metadata.
  - Profiling is split into two subprocesses, not one. ncu profiles a
    process's *entire* lifetime by default -- there is no implicit "just the
    last call" boundary -- so generating inputs (RNG kernels) and warming up
    Triton's JIT/autotune cache must happen in a separate, unprofiled phase 1
    process first; phase 2 (run under ncu) only ever loads the exact tensors
    phase 1 produced and makes one forward+backward call, so the captured
    report contains nothing but the candidate's own kernels from that call.
  - A single forward+backward call here can dispatch several Triton kernels
    (this benchmark's naive seed launches 7 for backward alone), so every
    kernel action in the captured report is collected and aggregated -- not
    just action 0, which is all the reference skill's tooling ever assumes.

Untested against a real ncu/ncu_report install (no GPU in this environment).
Smoke-test with the __main__ CLI below before relying on it in an evaluator.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parents[1]

# Kept short on purpose: this list is collected with `--metrics` (1-2 replay
# passes), not `--set full` (~45 passes) -- the per-evaluation ncu cost has to
# stay small enough to run inside OpenEvolve's evaluator timeout.
#
# Every name here exists because classify_patterns() below thresholds it
# against a documented cutoff from ncu-report-skill/reference/06-diagnosis-
# playbook.md (Patterns A, E, J, K -- the ones that don't require --set
# source/PM-sampling). Don't add a metric without also wiring it into a
# pattern check, and don't add a pattern check without an exact source.
DEFAULT_METRICS: tuple[str, ...] = (
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    # Pattern E's documented DRAM threshold is on this exact metric, not the
    # dram__throughput one above (kept too -- it's still informative, just not
    # the one the skill thresholds).
    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",  # achieved occupancy (Pattern J)
    "sm__maximum_warps_per_active_cycle_pct",  # theoretical occupancy (Pattern J)
    "launch__registers_per_thread",  # Pattern K
    "launch__waves_per_multiprocessor",  # Pattern A
    "launch__grid_size",  # Pattern A
    "device__attribute_multiprocessor_count",  # Pattern A (device constant, not per-kernel)
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",  # Pattern E
    "smsp__sass_inst_executed_op_local_ld.sum",  # Pattern K
    "smsp__sass_inst_executed_op_local_st.sum",  # Pattern K
)

_IMPORT_PREAMBLE = textwrap.dedent(
    """\
    import importlib.util
    import sys
    from pathlib import Path

    import torch

    REPO_ROOT = Path({repo_root!r})
    BENCH_DIR = Path({bench_dir!r})
    for _p in (REPO_ROOT, BENCH_DIR):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

    import task_spec

    spec = importlib.util.spec_from_file_location("ncu_profile_candidate", {program_path!r})
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    forward_fn = getattr(module, "layernorm_forward_with_saved")
    backward_fn = getattr(module, "layernorm_backward_from_saved")

    case = task_spec.TestCase({rows}, {cols}, {dtype_name!r}, {atol}, {rtol})
    """
)

# Phase 1, run WITHOUT ncu: generates inputs (the RNG kernels behind that must
# not show up in the profile) and runs `warmup` iterations so Triton's JIT
# compilation (and any autotuning) lands in its on-disk cache (~/.triton/cache
# by default, shared across processes) before ncu ever attaches. The exact
# input tensors are then saved so phase 2 reuses them byte-for-byte rather
# than regenerating (and re-launching RNG kernels for) them.
_WARMUP_TEMPLATE = _IMPORT_PREAMBLE + textwrap.dedent(
    """\
    torch.manual_seed(task_spec.seed_for_case(case))
    dy, x, weight, bias, eps = task_spec.make_inputs(torch, case)

    for _ in range({warmup}):
        y, saved = forward_fn(x, weight, bias, eps)
        backward_fn(dy, saved, eps)
    torch.cuda.synchronize()

    torch.save({{"dy": dy, "x": x, "weight": weight, "bias": bias, "eps": eps}}, {inputs_path!r})
    """
)

# Phase 2, run UNDER ncu: loads the exact inputs phase 1 produced and makes
# exactly one forward+backward call -- no RNG kernels, no repeated warmup
# calls, nothing else in this process's lifetime for ncu to capture besides
# the candidate's own kernels (ncu profiles the whole process by default,
# there is no implicit "just the last call" boundary).
_PROFILED_TEMPLATE = _IMPORT_PREAMBLE + textwrap.dedent(
    """\
    saved_inputs = torch.load({inputs_path!r}, map_location="cuda")
    dy, x, weight, bias, eps = (
        saved_inputs["dy"],
        saved_inputs["x"],
        saved_inputs["weight"],
        saved_inputs["bias"],
        saved_inputs["eps"],
    )

    y, saved = forward_fn(x, weight, bias, eps)
    backward_fn(dy, saved, eps)
    torch.cuda.synchronize()
    """
)


@dataclass
class KernelProfile:
    name: str
    duration_ns: float | None
    metrics: dict[str, Any]
    top_rule: dict[str, Any] | None


@dataclass
class ProfileResult:
    ok: bool
    error: str | None = None
    kernels: list[KernelProfile] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)
    top_rule: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "error": self.error,
            "aggregate": self.aggregate,
            "top_rule": self.top_rule,
            "kernels": [
                {"name": k.name, "duration_ns": k.duration_ns, "metrics": k.metrics, "top_rule": k.top_rule}
                for k in self.kernels
            ],
        }


def _ncu_report_dir_next_to_binary() -> Path | None:
    """Derive the extras/python dir from the `ncu` binary actually on PATH.

    Preferred over a blind glob: it's version-matched to whichever ncu will
    actually produce the .ncu-rep file, which matters since the binary report
    format can change between Nsight Compute releases.
    """
    import shutil

    ncu_bin = shutil.which("ncu")
    if not ncu_bin:
        return None
    # .../<install_root>/<version-or-not>/.../ncu -> look for extras/python
    # at a few plausible depths above the binary (HPC SDK nests it under
    # profilers/Nsight_Compute/, plain Nsight Compute installs don't).
    p = Path(ncu_bin).resolve()
    for ancestor in (p.parent, p.parent.parent, p.parent.parent.parent):
        candidate = ancestor / "extras" / "python"
        if (candidate / "ncu_report.py").exists():
            return candidate
    return None


def _find_ncu_report_module():
    """Locate the ncu_report module shipped with Nsight Compute.

    Tries a plain import first, then NCU_PYTHON_PATH, then the install next to
    the `ncu` binary on PATH (version-matched), then a glob under common CUDA
    /HPC-SDK install roots -- same fallback strategy the ncu-report-skill
    helpers use, since ncu_report ships outside any package index.
    """
    try:
        return importlib.import_module("ncu_report")
    except ImportError:
        pass

    candidates: list[Path] = []
    env_path = os.environ.get("NCU_PYTHON_PATH")
    if env_path:
        candidates.append(Path(env_path))

    beside_binary = _ncu_report_dir_next_to_binary()
    if beside_binary:
        candidates.append(beside_binary)

    for root in ("/usr/local", "/opt", "/opt/nvidia", "/opt/cuda"):
        p = Path(root)
        if not p.is_dir():
            continue
        candidates.extend(p.glob("cuda*/nsight-compute-*/extras/python"))
        candidates.extend(p.glob("nsight-compute-*/extras/python"))
        candidates.extend(p.glob("nsight-compute/*/extras/python"))
        # HPC SDK layout: <root>/hpc_sdk/<arch>/<version>/profilers/Nsight_Compute/extras/python
        candidates.extend(p.glob("hpc_sdk/*/*/profilers/Nsight_Compute/extras/python"))

    for c in candidates:
        if c.is_dir() and str(c) not in sys.path:
            sys.path.insert(0, str(c))
            try:
                return importlib.import_module("ncu_report")
            except ImportError:
                continue

    raise RuntimeError(
        "Could not import ncu_report. Set NCU_PYTHON_PATH to the directory "
        "containing ncu_report.py (under <cuda>/nsight-compute-*/extras/python, "
        "or <hpc_sdk>/profilers/Nsight_Compute/extras/python)."
    )


def _iter_indexed(obj, by_idx_name: str, count_name: str) -> Iterator[Any]:
    """Yield obj.<by_idx_name>(i) for i = 0, 1, ...

    Tries obj.<count_name>() first. The ncu_report binary module's exact API
    surface varies across Nsight Compute releases and isn't documented for the
    multi-action case (the ncu-report-skill this is adapted from only ever
    reads index 0, assuming `-c 1`), so this falls back to probing indices
    until access fails rather than assuming a method name exists.
    """
    by_idx = getattr(obj, by_idx_name)
    if hasattr(obj, count_name):
        try:
            n = getattr(obj, count_name)()
            for i in range(n):
                yield by_idx(i)
            return
        except Exception:
            pass
    i = 0
    while True:
        try:
            item = by_idx(i)
        except Exception:
            return
        if item is None:
            return
        yield item
        i += 1


def _write_warmup_script(tmp_dir: Path, program_path: str, case, warmup: int, inputs_path: Path) -> Path:
    content = _WARMUP_TEMPLATE.format(
        repo_root=str(REPO_ROOT),
        bench_dir=str(BENCHMARK_DIR),
        program_path=str(Path(program_path).resolve()),
        rows=case.rows,
        cols=case.cols,
        dtype_name=case.dtype_name,
        atol=case.atol_value,
        rtol=case.rtol_value,
        warmup=warmup,
        inputs_path=str(inputs_path),
    )
    script_path = tmp_dir / "warmup.py"
    script_path.write_text(content, encoding="utf-8")
    return script_path


def _write_profiled_script(tmp_dir: Path, program_path: str, case, inputs_path: Path) -> Path:
    content = _PROFILED_TEMPLATE.format(
        repo_root=str(REPO_ROOT),
        bench_dir=str(BENCHMARK_DIR),
        program_path=str(Path(program_path).resolve()),
        rows=case.rows,
        cols=case.cols,
        dtype_name=case.dtype_name,
        atol=case.atol_value,
        rtol=case.rtol_value,
        inputs_path=str(inputs_path),
    )
    script_path = tmp_dir / "profiled.py"
    script_path.write_text(content, encoding="utf-8")
    return script_path


def _safe_value(action, name: str):
    try:
        return action[name].value()
    except Exception:
        return None


def run_ncu_profile(
    program_path: str,
    case,
    *,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    warmup: int = 5,
    timeout: int = 120,
    ncu_bin: str = "ncu",
    python_bin: str | None = None,
) -> ProfileResult:
    """Profile one forward+backward call of `program_path` at `case` with ncu.

    Best-effort: any failure (ncu missing, no perf-counter permission, timeout,
    parse error) returns ProfileResult(ok=False, error=...) rather than raising,
    so a caller embedding this in an evaluator can fall back to scoring without
    hardware metrics instead of crashing the whole evaluation.
    """
    # NCU_PROFILE_PYTHON lets the harness use a specific interpreter (e.g. the
    # repo's venv) regardless of which python launched this process -- the ncu
    # subprocess otherwise inherits whatever sys.executable is, which silently
    # lacks torch if the caller's shell hasn't activated the right venv.
    python_bin = python_bin or os.environ.get("NCU_PROFILE_PYTHON") or sys.executable
    try:
        ncu_report = _find_ncu_report_module()
    except RuntimeError as exc:
        return ProfileResult(ok=False, error=str(exc))

    with tempfile.TemporaryDirectory(prefix="ncu_profile_") as tmp:
        tmp_dir = Path(tmp)
        inputs_path = tmp_dir / "inputs.pt"

        # Phase 1 (no ncu): generate inputs, warm up Triton's JIT/autotune
        # cache, save the exact tensors to disk. Anything that happens here
        # (RNG kernels, repeated warmup launches) must never reach ncu.
        warmup_script = _write_warmup_script(tmp_dir, program_path, case, warmup, inputs_path)
        try:
            warmup_completed = subprocess.run(
                [python_bin, str(warmup_script)],
                cwd=str(BENCHMARK_DIR),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ProfileResult(ok=False, error=f"warmup phase timed out after {timeout}s")
        except FileNotFoundError:
            return ProfileResult(ok=False, error=f"'{python_bin}' not found")
        if warmup_completed.returncode != 0 or not inputs_path.exists():
            return ProfileResult(
                ok=False,
                error=f"warmup phase failed (code={warmup_completed.returncode}): {warmup_completed.stderr[-2000:]}",
            )

        # Phase 2 (under ncu): load the saved inputs and make exactly one
        # forward+backward call. No RNG kernels, no warmup launches -- ncu
        # profiling this process's whole lifetime now only sees the
        # candidate's own kernels from that one call.
        profiled_script = _write_profiled_script(tmp_dir, program_path, case, inputs_path)
        report_base = tmp_dir / "report"

        cmd = [
            ncu_bin,
            "--metrics",
            ",".join(metrics),
            "--target-processes",
            "all",
            "-o",
            str(report_base),
            python_bin,
            str(profiled_script),
        ]
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(BENCHMARK_DIR),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ProfileResult(ok=False, error=f"ncu timed out after {timeout}s")
        except FileNotFoundError:
            return ProfileResult(ok=False, error=f"'{ncu_bin}' not found on PATH")

        report_path = report_base.with_suffix(".ncu-rep")
        if completed.returncode != 0 or not report_path.exists():
            return ProfileResult(
                ok=False,
                error=f"ncu failed (code={completed.returncode}): {completed.stderr[-2000:]}",
            )

        try:
            report = ncu_report.load_report(str(report_path))
        except Exception as exc:
            return ProfileResult(ok=False, error=f"failed to load .ncu-rep: {exc}")

        kernels: list[KernelProfile] = []
        for rng in _iter_indexed(report, "range_by_idx", "num_ranges"):
            for action in _iter_indexed(rng, "action_by_idx", "num_actions"):
                kmetrics = {name: _safe_value(action, name) for name in metrics}
                duration = kmetrics.get("gpu__time_duration.sum")

                top_rule = None
                try:
                    for rule in action.rule_results_as_dicts():
                        speedup = rule.get("estimated_speedup_pct")
                        if speedup is None:
                            continue
                        if top_rule is None or speedup > top_rule.get("estimated_speedup_pct", -1):
                            top_rule = rule
                except Exception:
                    pass

                kernels.append(
                    KernelProfile(name=action.name(), duration_ns=duration, metrics=kmetrics, top_rule=top_rule)
                )

    if not kernels:
        return ProfileResult(ok=False, error="ncu produced no kernel actions")

    aggregate: dict[str, Any] = {"total_kernels": len(kernels)}
    total_time = sum(k.duration_ns for k in kernels if k.duration_ns)
    aggregate["total_duration_ns"] = total_time or None

    for name in metrics:
        if name == "gpu__time_duration.sum":
            continue
        weighted_sum, weight_sum, plain_sum, plain_count = 0.0, 0.0, 0.0, 0
        for k in kernels:
            v = k.metrics.get(name)
            if v is None:
                continue
            plain_sum += v
            plain_count += 1
            w = k.duration_ns or 0.0
            weighted_sum += v * w
            weight_sum += w
        if plain_count == 0:
            aggregate[name] = None
        elif weight_sum > 0:
            # Time-weighted average: a kernel that ran for longer should
            # dominate the aggregate stall/throughput reading more than a
            # microsecond-scale reduction kernel launched alongside it.
            aggregate[name] = weighted_sum / weight_sum
        else:
            aggregate[name] = plain_sum / plain_count

    overall_top_rule = None
    for k in kernels:
        if k.top_rule and (
            overall_top_rule is None
            or k.top_rule.get("estimated_speedup_pct", -1) > overall_top_rule.get("estimated_speedup_pct", -1)
        ):
            overall_top_rule = k.top_rule

    return ProfileResult(ok=True, kernels=kernels, aggregate=aggregate, top_rule=overall_top_rule)


def derive_flags(aggregate: dict[str, Any]) -> dict[str, Any]:
    """Turn raw aggregate metrics into a few scoring-friendly booleans/scalars.

    Kept separate from classify_patterns() below: this is just a flat view of
    the raw numbers (no thresholding), useful for logging/plotting everything
    regardless of whether a documented pattern fired.
    """
    local_ld = aggregate.get("smsp__sass_inst_executed_op_local_ld.sum") or 0
    local_st = aggregate.get("smsp__sass_inst_executed_op_local_st.sum") or 0
    return {
        "register_spilling": bool(local_ld > 0 or local_st > 0),
        "occupancy_pct": aggregate.get("sm__warps_active.avg.pct_of_peak_sustained_active"),
        "long_scoreboard_stall_ratio": aggregate.get(
            "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"
        ),
        "sm_throughput_pct": aggregate.get("sm__throughput.avg.pct_of_peak_sustained_elapsed"),
        "dram_throughput_pct": aggregate.get("dram__throughput.avg.pct_of_peak_sustained_elapsed"),
    }


def classify_patterns(aggregate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Apply ncu-report-skill's documented diagnosis-playbook thresholds.

    Only patterns checkable from a `--metrics` (no `--set source`, no PM
    sampling) pass are implemented here, each cited to its exact source:

      Pattern A (small grid / SM idle)      -- 06-diagnosis-playbook.md
      Pattern E (latency-bound)             -- 06-diagnosis-playbook.md
      "bandwidth-bound" (Dimension 6 reading)-- 05-analysis-dimensions.md
      Pattern J (occupancy gap)             -- 06-diagnosis-playbook.md
      Pattern K (register spill)            -- 06-diagnosis-playbook.md

    Patterns C, D, F, G, H, I, L, M, N are NOT implemented: C/D/H/N need
    per-PC source-level data (`--set source`), M needs a PM-sampling timeline,
    and F (tensor cores) / G (atomics) / L (fp64) don't apply to this kernel
    family (no matmul, no atomics, fp32-accumulated by design).

    Every entry carries the specific metric value(s) that backed the
    decision, not just the boolean -- ncu-report-skill's SKILL.md is explicit
    that citing the actual numbers is "the deliverable," not the label alone.
    """
    g = aggregate.get

    waves = g("launch__waves_per_multiprocessor")
    grid_size = g("launch__grid_size")
    sm_count = g("device__attribute_multiprocessor_count")
    small_grid = bool((waves is not None and waves < 0.5) or (grid_size is not None and sm_count is not None and grid_size < sm_count))
    small_grid_evidence = f"launch__waves_per_multiprocessor={waves}, launch__grid_size={grid_size}, device__attribute_multiprocessor_count={sm_count} (Pattern A: waves<0.5 or grid_size<sm_count)"

    long_scoreboard = g("smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio")
    dram_bytes_pct = g("dram__bytes_read.sum.pct_of_peak_sustained_elapsed")
    latency_bound = bool(
        long_scoreboard is not None and long_scoreboard > 3 and dram_bytes_pct is not None and dram_bytes_pct < 10
    )
    latency_bound_evidence = f"long_scoreboard_ratio={long_scoreboard}, dram_bytes_read_pct={dram_bytes_pct} (Pattern E: ratio>3 and dram_pct<10)"

    bandwidth_bound = bool(dram_bytes_pct is not None and dram_bytes_pct >= 80)
    bandwidth_bound_evidence = f"dram_bytes_read_pct={dram_bytes_pct} (Dimension 6: >=80 is genuinely bandwidth-bound)"

    theoretical_occ = g("sm__maximum_warps_per_active_cycle_pct")
    achieved_occ = g("sm__warps_active.avg.pct_of_peak_sustained_active")
    occupancy_gap = (
        (theoretical_occ - achieved_occ) if (theoretical_occ is not None and achieved_occ is not None) else None
    )
    occupancy_limited = bool(theoretical_occ is not None and achieved_occ is not None and theoretical_occ > 50 and achieved_occ < 50)
    occupancy_evidence = f"theoretical_occupancy_pct={theoretical_occ}, achieved_occupancy_pct={achieved_occ}, gap={occupancy_gap} (Pattern J: theoretical>50 and achieved<50)"

    local_ld = g("smsp__sass_inst_executed_op_local_ld.sum") or 0
    local_st = g("smsp__sass_inst_executed_op_local_st.sum") or 0
    regs = g("launch__registers_per_thread")
    register_spill = bool(local_ld > 0 or local_st > 0 or (regs is not None and regs > 128))
    register_spill_evidence = f"local_ld={local_ld}, local_st={local_st}, registers_per_thread={regs} (Pattern K: ld/st>0 or registers>128)"

    return {
        "small_grid": {"flag": small_grid, "pattern": "A - small grid / SM idle", "evidence": small_grid_evidence},
        "latency_bound": {"flag": latency_bound, "pattern": "E - latency-bound", "evidence": latency_bound_evidence},
        "bandwidth_bound": {
            "flag": bandwidth_bound,
            "pattern": "Dimension 6 - bandwidth-bound",
            "evidence": bandwidth_bound_evidence,
        },
        "occupancy_limited": {
            "flag": occupancy_limited,
            "pattern": "J - low achieved vs theoretical occupancy",
            "evidence": occupancy_evidence,
        },
        "register_spill": {
            "flag": register_spill,
            "pattern": "K - register spill",
            "evidence": register_spill_evidence,
        },
    }


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        print(f"Usage: {argv[0]} PROGRAM_PATH ROWS COLS DTYPE_NAME")
        return 2
    program_path, rows, cols, dtype_name = argv[1:5]

    sys.path.insert(0, str(BENCHMARK_DIR))
    import task_spec  # noqa: E402

    case = task_spec.TestCase(int(rows), int(cols), dtype_name, 5e-2, 5e-2)
    result = run_ncu_profile(program_path, case)
    print(json.dumps(result.to_json(), indent=2, sort_keys=True, default=str))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
