"""Unified correctness smoke tests for AtenIR-backed backends."""

from __future__ import annotations

import pytest
import torch

from tests.atenir_correctness.harness import (
    AtenIRComposeBackend,
    ShapeCase,
    check_backend_against_autograd,
    require_cuda_or_skip,
)


ATENIR_EXAMPLE_CASES = [
    ("square_sum", "atenir._examples:square_sum", ((16, 17),), 1e-4, 1e-4),
    ("rmsnorm", "atenir._examples:rmsnorm", ((16, 64), (64,)), 2e-4, 2e-4),
    ("layernorm", "atenir._examples:layernorm", ((8, 64), (64,), (64,)), 2e-5, 2e-5),
    ("attention_block", "atenir._examples:attention_block", ((2, 8, 16), (2, 8, 16), (2, 8, 16)), 2e-4, 2e-4),
    ("topk_gather", "atenir._examples:topk_gather", ((16, 32),), 1e-4, 1e-4),
]


@pytest.mark.parametrize(
    "name,fn_spec,shapes,atol,rtol",
    ATENIR_EXAMPLE_CASES,
)
def test_atenir_compose_examples_smoke(name, fn_spec, shapes, atol, rtol):
    require_cuda_or_skip()
    case = ShapeCase(f"{name}_smoke", shapes, atol=atol, rtol=rtol, seed=11)
    backend = AtenIRComposeBackend(f"atenir_compose_{name}", fn_spec)
    reports = check_backend_against_autograd(backend, fn_spec, case)
    assert all(report.passed for report in reports)
    assert backend.fallback_report["total"] >= 1


@pytest.mark.parametrize(
    "case",
    [
        ShapeCase("square_sum_static", ((16, 17),), atol=1e-4, rtol=1e-4, seed=11),
        ShapeCase("square_sum_nontile", ((7, 31),), atol=1e-4, rtol=1e-4, seed=12),
    ],
)
def test_atenir_compose_square_sum(case: ShapeCase):
    require_cuda_or_skip()
    backend = AtenIRComposeBackend("atenir_compose_square_sum", "atenir._examples:square_sum")
    reports = check_backend_against_autograd(backend, "atenir._examples:square_sum", case)
    assert all(report.passed for report in reports)
    assert backend.fallback_report["total"] >= 1


def test_shape_case_can_use_fp16_metadata():
    case = ShapeCase("metadata_only", ((2, 8),), dtype=torch.float16, atol=1e-2, rtol=1e-2)
    assert case.dtype is torch.float16
