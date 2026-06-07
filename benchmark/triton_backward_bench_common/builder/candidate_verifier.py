"""Candidate verifier for Stage 1 generated naive Triton backward files."""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any


def syntax_check_command(task_dir: Path, python: str = "python") -> str:
    return (
        f"{python} - <<'PY'\n"
        "from pathlib import Path\n"
        f"root = Path({str(task_dir)!r})\n"
        "for path in sorted(root.glob('*.py')):\n"
        "    compile(path.read_text(), str(path), 'exec')\n"
        "    print(f'compiled {path.name}')\n"
        "PY"
    )


def verification_commands(task_dir: Path, python: str = "python") -> list[str]:
    return [
        syntax_check_command(task_dir, python),
        f"cd {task_dir} && {python} test_correctness.py",
        f"cd {task_dir} && {python} evaluator.py initial_program.py",
    ]


def non_gpu_commands(task_dir: Path, python: str = "python") -> list[str]:
    return [syntax_check_command(task_dir, python)]


def _load_module(path: Path, prefix: str):
    module_name = f"{prefix}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_outputs(output: Any, output_names: tuple[str, ...]):
    if len(output_names) == 1:
        return (output,)
    if not isinstance(output, tuple) or len(output) != len(output_names):
        expected = ", ".join(output_names)
        raise TypeError(f"candidate must return tuple ({expected})")
    return output


def _max_errors(torch_module, actual, expected):
    diff = (actual.float() - expected.float()).abs()
    max_abs = float(torch_module.max(diff).item())
    denom = torch_module.clamp(expected.float().abs(), min=1e-8)
    max_rel = float(torch_module.max(diff / denom).item())
    return max_abs, max_rel


def verify_candidate(
    task_dir: Path,
    candidate_path: Path,
    candidate_fn_name: str,
    max_cases: int | None = None,
) -> dict[str, Any]:
    examples_dir = task_dir.parent
    if str(task_dir) not in sys.path:
        sys.path.insert(0, str(task_dir))
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))

    try:
        import torch
        import task_spec
        import triton  # noqa: F401
    except Exception as exc:
        return {
            "passed": False,
            "error_type": "ImportError",
            "error_message": str(exc),
            "traceback": traceback.format_exc(limit=8),
        }

    if not torch.cuda.is_available():
        return {
            "passed": False,
            "error_type": "RuntimeUnavailable",
            "error_message": "CUDA is not available; run Stage 1 candidate verification on a GPU node",
        }

    try:
        candidate_module = _load_module(candidate_path, "stage1_candidate")
        candidate_fn = getattr(candidate_module, candidate_fn_name)
    except Exception as exc:
        return {
            "passed": False,
            "error_type": "CandidateLoadError",
            "error_message": str(exc),
            "traceback": traceback.format_exc(limit=8),
        }

    reports = []
    passed_cases = 0
    cases = task_spec.CORRECTNESS_CASES[:max_cases] if max_cases else task_spec.CORRECTNESS_CASES
    output_names = tuple(task_spec.OUTPUT_NAMES)

    for case in cases:
        case_report = task_spec.case_metadata(case) if hasattr(task_spec, "case_metadata") else {"case": repr(case)}
        try:
            if hasattr(task_spec, "seed_for_case"):
                torch.manual_seed(task_spec.seed_for_case(case))
            inputs = task_spec.make_inputs(torch, case)
            expected = _normalize_outputs(task_spec.torch_oracle(torch, *inputs), output_names)
            actual = _normalize_outputs(candidate_fn(*inputs), output_names)
            torch.cuda.synchronize()

            report = {**case_report, "correct": True}
            for name, actual_tensor, expected_tensor in zip(output_names, actual, expected):
                if actual_tensor.shape != expected_tensor.shape:
                    report["correct"] = False
                    report[f"{name}_correct"] = False
                    report[f"{name}_error_type"] = "ShapeMismatch"
                    report[f"{name}_actual_shape"] = list(actual_tensor.shape)
                    report[f"{name}_expected_shape"] = list(expected_tensor.shape)
                    continue
                atol = task_spec.atol(case, name)
                rtol = task_spec.rtol(case, name)
                max_abs, max_rel = _max_errors(torch, actual_tensor, expected_tensor)
                is_correct = bool(torch.allclose(actual_tensor, expected_tensor, atol=atol, rtol=rtol))
                report[f"{name}_correct"] = is_correct
                report[f"{name}_max_abs_error"] = max_abs
                report[f"{name}_max_rel_error"] = max_rel
                report[f"{name}_atol"] = atol
                report[f"{name}_rtol"] = rtol
                report["correct"] = report["correct"] and is_correct
            passed_cases += int(report["correct"])
            reports.append(report)
        except Exception as exc:
            reports.append(
                {
                    **case_report,
                    "correct": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                }
            )

    total = len(cases)
    return {
        "passed": passed_cases == total,
        "passed_cases": passed_cases,
        "total_cases": total,
        "candidate_path": str(candidate_path),
        "candidate_fn_name": candidate_fn_name,
        "reports": reports,
    }


def main(argv: list[str]) -> int:
    if len(argv) < 4:
        print("Usage: candidate_verifier.py TASK_DIR CANDIDATE_PATH CANDIDATE_FN_NAME [MAX_CASES]")
        return 2
    task_dir = Path(argv[1]).resolve()
    candidate_path = Path(argv[2]).resolve()
    candidate_fn_name = argv[3]
    max_cases = int(argv[4]) if len(argv) > 4 else None
    report = verify_candidate(task_dir, candidate_path, candidate_fn_name, max_cases)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
