"""Stage 1 API-driven naive Triton backward synthesis loop."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .llm_client import generate_with_openai_compatible_api
from .prompt_templates import (
    render_backward_formula_prompt,
    render_naive_triton_prompt,
    render_repair_prompt_template,
)
from .spec import load_benchmark_spec


SYSTEM_MESSAGE = """You are an expert Triton GPU programmer and numerical autodiff engineer.
You write naive, readable, correctness-first Triton backward baselines.
The ground truth is PyTorch autograd over forward_ref.py, not your derivation.
For Stage 1, prefer faithful unfused decomposition over performance."""


@dataclass(frozen=True)
class SynthesizeConfig:
    task_dir: Path
    output_dir: Path
    api_base: str
    model: str
    api_key: str | None
    max_attempts: int
    max_tokens: int
    temperature: float | None
    timeout: int
    candidate_fn_name: str
    python: str
    max_cases: int | None
    dry_run: bool = False


def infer_candidate_fn_name(operator: str) -> str:
    return f"{operator}_backward_naive_triton"


def _read_if_exists(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _strip_code_fence(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip() + "\n"
    return text.strip() + "\n"


def _infer_make_inputs_order(task_dir: Path) -> list[str] | None:
    task_spec_path = task_dir / "task_spec.py"
    source = _read_if_exists(task_spec_path)
    match = re.search(r"def make_inputs\(.*?\):(?P<body>.*?)(?:\n\n|\Z)", source, flags=re.DOTALL)
    if not match:
        return None
    returns = re.findall(r"return\s+([^\n]+)", match.group("body"))
    if not returns:
        return None
    return [part.strip() for part in returns[-1].split(",")]


def _signature_for_prompt(config: SynthesizeConfig) -> str:
    spec = load_benchmark_spec(config.task_dir)
    ordered_args = _infer_make_inputs_order(config.task_dir)
    if not ordered_args:
        ordered_args = [spec.upstream_gradient] + [input_spec.name for input_spec in spec.inputs]
    arg_lines = []
    for arg in ordered_args:
        if arg == "EPS":
            arg_lines.append("    eps: float = 1e-5,")
        elif arg in {"eps", "epsilon"}:
            arg_lines.append(f"    {arg}: float = 1e-5,")
        else:
            arg_lines.append(f"    {arg}: torch.Tensor,")
    returns = ", ".join(["torch.Tensor"] * len(spec.required_gradients))
    return (
        f"def {config.candidate_fn_name}(\n"
        + "\n".join(arg_lines)
        + f"\n) -> tuple[{returns}]:"
    )


def _call_order_for_prompt(config: SynthesizeConfig) -> str:
    ordered_args = _infer_make_inputs_order(config.task_dir)
    if not ordered_args:
        return f"{config.candidate_fn_name}(*inputs)"
    args = ", ".join(f"inputs[{index}]" for index in range(len(ordered_args)))
    return f"{config.candidate_fn_name}({args})"


def _context_block(config: SynthesizeConfig) -> str:
    stage1_spec = _read_if_exists(config.task_dir / "stage1_spec.yaml")
    forward_ref = _read_if_exists(config.task_dir / "forward_ref.py")
    forward_triton = _read_if_exists(config.task_dir / "forward_triton.py")
    return f"""
Stage 1 normalized spec:

```yaml
{stage1_spec}
```

forward_ref.py:

```python
{forward_ref}
```

forward_triton.py, provided only as layout/indexing context:

```python
{forward_triton}
```
"""


def _formula_prompt(config: SynthesizeConfig) -> str:
    spec = load_benchmark_spec(config.task_dir)
    return f"""{render_backward_formula_prompt(spec)}

{_context_block(config)}

Return only Markdown describing the naive backward decomposition. Do not write code.
"""


def _base_prompt(config: SynthesizeConfig, decomposition: str) -> str:
    spec = load_benchmark_spec(config.task_dir)
    prompt = render_naive_triton_prompt(spec)
    signature = _signature_for_prompt(config)
    call_order = _call_order_for_prompt(config)
    return f"""{prompt}

Public API requirement:

```python
{signature}
    ...
```

The verifier calls this exact function as:

```python
inputs = task_spec.make_inputs(torch, case)
actual = {call_order}
```

You must implement exactly the signature shown above.
Do not use `*args`, `**kwargs`, argument-order inference, or compatibility
branches for alternate argument orders.

Naive decomposition to implement:

```markdown
{decomposition}
```

Implementation policy:

- Implement the decomposition above directly.
- Follow the canonical kernel plan from the prompt exactly when present.
- Prefer one small Triton kernel per named intermediate/reduction group.
- Materialize intermediate tensors where the decomposition names them.
- Keep comments short and focused on the decomposition.

{_context_block(config)}

Return only the complete Python source file. Do not include Markdown prose.
"""


def _repair_prompt(config: SynthesizeConfig, previous_code: str, verifier_report: str) -> str:
    spec = load_benchmark_spec(config.task_dir)
    repair_template = render_repair_prompt_template(spec)
    decomposition = _read_if_exists(config.output_dir / "naive_decomposition.md")
    signature = _signature_for_prompt(config)
    return f"""{repair_template.replace("{VERIFIER_ERROR_REPORT}", verifier_report)}

Required public API signature:

```python
{signature}
    ...
```

Repair must preserve this exact signature. Do not use `*args`, `**kwargs`, or
argument-order compatibility branches.

Naive decomposition that must remain the implementation guide:

```markdown
{decomposition}
```

Previous candidate source:

```python
{previous_code}
```

Return only the repaired complete Python source file. Do not include Markdown prose.
"""


def _write_attempt_inputs(attempt_dir: Path, prompt: str, response: str, code: str) -> Path:
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    (attempt_dir / "response.txt").write_text(response, encoding="utf-8")
    candidate_path = attempt_dir / "backward_naive_triton_candidate.py"
    candidate_path.write_text(code, encoding="utf-8")
    return candidate_path


def _verify_candidate(config: SynthesizeConfig, candidate_path: Path) -> dict:
    cmd = [
        config.python,
        "-m",
        "benchmark.triton_backward_bench_common.builder.candidate_verifier",
        str(config.task_dir),
        str(candidate_path),
        config.candidate_fn_name,
    ]
    if config.max_cases is not None:
        cmd.append(str(config.max_cases))
    completed = subprocess.run(
        cmd,
        cwd=str(config.task_dir.parents[1]),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError:
        report = {
            "passed": False,
            "error_type": "VerifierInvocationError",
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "returncode": completed.returncode,
        }
    if completed.stderr:
        report["stderr"] = completed.stderr
    return report


def synthesize(config: SynthesizeConfig) -> int:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    formula_prompt = _formula_prompt(config)
    if config.dry_run:
        dryrun_dir = config.output_dir / "attempt_001"
        dryrun_dir.mkdir(parents=True, exist_ok=True)
        (config.output_dir / "formula_prompt.md").write_text(formula_prompt, encoding="utf-8")
        (dryrun_dir / "prompt.md").write_text(
            _base_prompt(config, "{NAIVE_DECOMPOSITION_FROM_FORMULA_STAGE}"),
            encoding="utf-8",
        )
        print(f"dry-run wrote {config.output_dir / 'formula_prompt.md'}")
        print(f"dry-run wrote {dryrun_dir / 'prompt.md'}")
        return 0

    print("Stage 1 formula/decomposition synthesis")
    decomposition = generate_with_openai_compatible_api(
        prompt=formula_prompt,
        system_message=SYSTEM_MESSAGE,
        model=config.model,
        api_base=config.api_base,
        api_key=config.api_key,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        timeout=config.timeout,
    )
    (config.output_dir / "formula_prompt.md").write_text(formula_prompt, encoding="utf-8")
    (config.output_dir / "naive_decomposition.md").write_text(decomposition, encoding="utf-8")
    prompt = _base_prompt(config, decomposition)
    previous_code = ""

    for attempt in range(1, config.max_attempts + 1):
        attempt_dir = config.output_dir / f"attempt_{attempt:03d}"
        print(f"Stage 1 synthesis attempt {attempt}/{config.max_attempts}")
        response = generate_with_openai_compatible_api(
            prompt=prompt,
            system_message=SYSTEM_MESSAGE,
            model=config.model,
            api_base=config.api_base,
            api_key=config.api_key,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout=config.timeout,
        )
        code = _strip_code_fence(response)
        candidate_path = _write_attempt_inputs(attempt_dir, prompt, response, code)
        report = _verify_candidate(config, candidate_path)
        (attempt_dir / "verification_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        if report.get("passed"):
            best_dir = config.output_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            best_path = best_dir / "backward_naive_triton_candidate.py"
            best_path.write_text(code, encoding="utf-8")
            (best_dir / "verification_report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(f"Stage 1 synthesis passed. Best candidate: {best_path}")
            return 0

        verifier_report = json.dumps(report, indent=2, sort_keys=True)
        (attempt_dir / "repair_prompt.md").write_text(
            _repair_prompt(config, code, verifier_report),
            encoding="utf-8",
        )
        previous_code = code
        prompt = _repair_prompt(config, previous_code, verifier_report)
        print(f"attempt {attempt} failed; wrote repair prompt to {attempt_dir / 'repair_prompt.md'}")

    print(f"Stage 1 synthesis failed after {config.max_attempts} attempts")
    return 1
