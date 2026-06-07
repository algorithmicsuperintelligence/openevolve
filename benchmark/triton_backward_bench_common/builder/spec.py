"""Dataclasses for Stage 1 Triton backward benchmark construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class InputSpec:
    name: str
    shape: str
    layout: str = "unspecified"
    differentiable: bool = True


@dataclass(frozen=True)
class WorkloadSpec:
    correctness_shapes: list[list[int]]
    correctness_dtypes: list[str]
    benchmark_shapes: list[list[int]]
    benchmark_dtypes: list[str]


@dataclass(frozen=True)
class BenchmarkSpec:
    task_name: str
    operator: str
    difficulty_level: int
    difficulty_label: str
    description: str
    forward_formula: str
    inputs: list[InputSpec]
    output_shape: str
    upstream_gradient: str
    required_gradients: list[str]
    backward_formulas: dict[str, str]
    saved_tensors: list[str]
    workloads: WorkloadSpec
    tolerances: dict[str, dict[str, float]]
    baselines: dict[str, Any] = field(default_factory=dict)
    metrics: list[str] = field(default_factory=list)

    def differentiable_inputs(self) -> list[InputSpec]:
        return [input_spec for input_spec in self.inputs if input_spec.differentiable]

    def to_stage1_dict(self) -> dict[str, Any]:
        return {
            "stage": "stage1_forward_to_backward_construction",
            "task_name": self.task_name,
            "operator": self.operator,
            "difficulty": {
                "level": self.difficulty_level,
                "label": self.difficulty_label,
            },
            "description": self.description,
            "semantic_source": {
                "forward_ref": "forward_ref.py",
                "forward_formula": self.forward_formula,
            },
            "inputs": [
                {
                    "name": input_spec.name,
                    "shape": input_spec.shape,
                    "layout": input_spec.layout,
                    "differentiable": input_spec.differentiable,
                }
                for input_spec in self.inputs
            ],
            "output": {
                "name": "y",
                "shape": self.output_shape,
            },
            "backward_target": {
                "upstream_gradient": self.upstream_gradient,
                "required_gradients": self.required_gradients,
                "saved_tensors": self.saved_tensors,
                "formula_hints": self.backward_formulas,
            },
            "oracle_policy": {
                "ground_truth": "PyTorch autograd over forward_ref.py",
                "llm_generated_code_is_not_ground_truth": True,
            },
            "workloads": {
                "correctness": {
                    "shapes": self.workloads.correctness_shapes,
                    "dtypes": self.workloads.correctness_dtypes,
                },
                "benchmark": {
                    "shapes": self.workloads.benchmark_shapes,
                    "dtypes": self.workloads.benchmark_dtypes,
                },
            },
            "tolerances": self.tolerances,
            "baselines": self.baselines,
            "metrics": self.metrics,
            "construction_outputs": {
                "autograd_oracle": "backward_ref.py",
                "naive_triton_candidate": "backward_naive_triton.py",
                "prompts": [
                    "prompts/backward_formula_prompt.md",
                    "prompts/naive_triton_backward_prompt.md",
                    "prompts/repair_prompt_template.md",
                ],
            },
        }


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def load_benchmark_spec(task_dir: Path) -> BenchmarkSpec:
    data = load_yaml(task_dir / "meta.yaml")

    forward = data.get("forward", {})
    backward = data.get("backward", {})
    workloads = data.get("workloads", {})
    correctness = workloads.get("correctness", {})
    benchmark = workloads.get("benchmark", {})

    saved_tensors = set(backward.get("saved_tensors", []))
    required_gradients = list(backward.get("required_gradients", []))
    inputs = []
    for name, input_data in forward.get("inputs", {}).items():
        grad_names = {f"d{name}", f"d{name.removesuffix('_ref')}"}
        differentiable = name in saved_tensors or bool(grad_names.intersection(required_gradients))
        inputs.append(
            InputSpec(
                name=name,
                shape=str(input_data.get("shape", "unspecified")),
                layout=str(input_data.get("layout", "unspecified")),
                differentiable=differentiable,
            )
        )

    return BenchmarkSpec(
        task_name=str(data["task_name"]),
        operator=str(data["operator"]),
        difficulty_level=int(data.get("difficulty_level", 0)),
        difficulty_label=str(data.get("difficulty_label", "unspecified")),
        description=str(data.get("description", "")).strip(),
        forward_formula=str(forward.get("formula", "")),
        inputs=inputs,
        output_shape=str(forward.get("output", {}).get("y", {}).get("shape", "unspecified")),
        upstream_gradient=str(backward.get("upstream_gradient", "dy")),
        required_gradients=required_gradients,
        backward_formulas=dict(backward.get("formulas", {})),
        saved_tensors=list(backward.get("saved_tensors", [])),
        workloads=WorkloadSpec(
            correctness_shapes=list(correctness.get("shapes", [])),
            correctness_dtypes=list(correctness.get("dtypes", [])),
            benchmark_shapes=list(benchmark.get("shapes", [])),
            benchmark_dtypes=list(benchmark.get("dtypes", [])),
        ),
        tolerances=dict(data.get("tolerances", {})),
        baselines=dict(data.get("baselines", {})),
        metrics=list(data.get("metrics", [])),
    )
