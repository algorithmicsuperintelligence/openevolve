"""Triton-native ncu profiling for the EvoformerAttention autograd-pair benchmark.

Structurally identical to the LayerNorm ncu_profile.py; the only operator-
specific parts are the warmup/profile script templates (different function
names, different input layout) and the main() CLI argument list.
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

DEFAULT_METRICS: tuple[str, ...] = (
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__maximum_warps_per_active_cycle_pct",
    "launch__registers_per_thread",
    "launch__waves_per_multiprocessor",
    "launch__grid_size",
    "device__attribute_multiprocessor_count",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__sass_inst_executed_op_local_ld.sum",
    "smsp__sass_inst_executed_op_local_st.sum",
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

    forward_fn = getattr(module, "evoattention_forward_with_saved")
    backward_fn = getattr(module, "evoattention_backward_from_saved")

    case = task_spec.TestCase({b}, {n_seq}, {head}, {n_res}, {dim}, {dtype_name!r}, {atol}, {rtol})
    """
)

_WARMUP_TEMPLATE = _IMPORT_PREAMBLE + textwrap.dedent(
    """\
    torch.manual_seed(task_spec.seed_for_case(case))
    do, q, k, v, res_mask, pair_bias = task_spec.make_inputs(torch, case)

    for _ in range({warmup}):
        y, saved = forward_fn(q, k, v, res_mask, pair_bias)
        backward_fn(do, saved)
    torch.cuda.synchronize()

    torch.save(
        {{"do": do, "q": q, "k": k, "v": v, "res_mask": res_mask, "pair_bias": pair_bias}},
        {inputs_path!r},
    )
    """
)

_PROFILED_TEMPLATE = _IMPORT_PREAMBLE + textwrap.dedent(
    """\
    saved_inputs = torch.load({inputs_path!r}, map_location="cuda")
    do        = saved_inputs["do"]
    q         = saved_inputs["q"]
    k         = saved_inputs["k"]
    v         = saved_inputs["v"]
    res_mask  = saved_inputs["res_mask"]
    pair_bias = saved_inputs["pair_bias"]

    y, saved = forward_fn(q, k, v, res_mask, pair_bias)
    backward_fn(do, saved)
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
    import shutil

    ncu_bin = shutil.which("ncu")
    if not ncu_bin:
        return None
    p = Path(ncu_bin).resolve()
    for ancestor in (p.parent, p.parent.parent, p.parent.parent.parent):
        candidate = ancestor / "extras" / "python"
        if (candidate / "ncu_report.py").exists():
            return candidate
    return None


def _find_ncu_report_module():
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
        "containing ncu_report.py (under <cuda>/nsight-compute-*/extras/python)."
    )


def _iter_indexed(obj, by_idx_name: str, count_name: str) -> Iterator[Any]:
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
        b=case.b,
        n_seq=case.n_seq,
        head=case.head,
        n_res=case.n_res,
        dim=case.dim,
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
        b=case.b,
        n_seq=case.n_seq,
        head=case.head,
        n_res=case.n_res,
        dim=case.dim,
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
    python_bin = python_bin or os.environ.get("NCU_PROFILE_PYTHON") or sys.executable
    try:
        ncu_report = _find_ncu_report_module()
    except RuntimeError as exc:
        return ProfileResult(ok=False, error=str(exc))

    with tempfile.TemporaryDirectory(prefix="ncu_profile_") as tmp:
        tmp_dir = Path(tmp)
        inputs_path = tmp_dir / "inputs.pt"

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
    g = aggregate.get

    waves = g("launch__waves_per_multiprocessor")
    grid_size = g("launch__grid_size")
    sm_count = g("device__attribute_multiprocessor_count")
    small_grid = bool(
        (waves is not None and waves < 0.5)
        or (grid_size is not None and sm_count is not None and grid_size < sm_count)
    )
    small_grid_evidence = (
        f"launch__waves_per_multiprocessor={waves}, launch__grid_size={grid_size}, "
        f"device__attribute_multiprocessor_count={sm_count} (Pattern A: waves<0.5 or grid_size<sm_count)"
    )

    long_scoreboard = g("smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio")
    dram_bytes_pct = g("dram__bytes_read.sum.pct_of_peak_sustained_elapsed")
    latency_bound = bool(
        long_scoreboard is not None and long_scoreboard > 3
        and dram_bytes_pct is not None and dram_bytes_pct < 10
    )
    latency_bound_evidence = (
        f"long_scoreboard_ratio={long_scoreboard}, dram_bytes_read_pct={dram_bytes_pct} "
        f"(Pattern E: ratio>3 and dram_pct<10)"
    )

    bandwidth_bound = bool(dram_bytes_pct is not None and dram_bytes_pct >= 80)
    bandwidth_bound_evidence = (
        f"dram_bytes_read_pct={dram_bytes_pct} (Dimension 6: >=80 is genuinely bandwidth-bound)"
    )

    theoretical_occ = g("sm__maximum_warps_per_active_cycle_pct")
    achieved_occ = g("sm__warps_active.avg.pct_of_peak_sustained_active")
    occupancy_gap = (
        (theoretical_occ - achieved_occ)
        if (theoretical_occ is not None and achieved_occ is not None)
        else None
    )
    occupancy_limited = bool(
        theoretical_occ is not None and achieved_occ is not None
        and theoretical_occ > 50 and achieved_occ < 50
    )
    occupancy_evidence = (
        f"theoretical_occupancy_pct={theoretical_occ}, achieved_occupancy_pct={achieved_occ}, "
        f"gap={occupancy_gap} (Pattern J: theoretical>50 and achieved<50)"
    )

    local_ld = g("smsp__sass_inst_executed_op_local_ld.sum") or 0
    local_st = g("smsp__sass_inst_executed_op_local_st.sum") or 0
    regs = g("launch__registers_per_thread")
    register_spill = bool(local_ld > 0 or local_st > 0 or (regs is not None and regs > 128))
    register_spill_evidence = (
        f"local_ld={local_ld}, local_st={local_st}, registers_per_thread={regs} "
        f"(Pattern K: ld/st>0 or registers>128)"
    )

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
    if len(argv) != 8:
        print(f"Usage: {argv[0]} PROGRAM_PATH B N_SEQ HEAD N_RES DIM DTYPE_NAME")
        return 2
    program_path, b, n_seq, head, n_res, dim, dtype_name = argv[1:]

    sys.path.insert(0, str(BENCHMARK_DIR))
    import task_spec  # noqa: E402

    case = task_spec.TestCase(int(b), int(n_seq), int(head), int(n_res), int(dim), dtype_name, 5e-2, 5e-2)
    result = run_ncu_profile(program_path, case)
    print(json.dumps(result.to_json(), indent=2, sort_keys=True, default=str))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
