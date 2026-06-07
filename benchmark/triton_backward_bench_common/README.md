# Triton Backward Benchmark Common

Shared utilities for forward-to-backward Triton benchmark examples.

## Pipeline

Stage 1 builds a verified naive backward baseline:

```text
meta.yaml + forward_ref.py
        |
        v
load task spec
        |
        v
generate formula/decomposition prompt
        |
        v
LLM writes naive_decomposition.md
        |
        v
generate Triton-code prompt
        |
        v
LLM writes backward_naive_triton_candidate.py
        |
        v
verifier compares against PyTorch autograd oracle
        |
        +-- pass: save to stage1_candidates/best
        |
        +-- fail: write repair prompt and retry
```

Stage 2 optimizes a verified naive backward baseline with OpenEvolve. The
evaluator keeps correctness as a hard gate: candidates are benchmarked only
after every required gradient matches the PyTorch-autograd oracle.

## Files

- `evaluator_core.py`: shared OpenEvolve evaluator. It imports a candidate,
  checks the public API, compares each gradient against `task_spec.torch_oracle`,
  and benchmarks only correct candidates.
- `builder_cli.py`: command-line entry point for Stage 1 artifacts and synthesis.
- `builder/spec.py`: reads `meta.yaml` and converts it to a normalized
  `BenchmarkSpec`.
- `builder/prompt_templates.py`: builds formula, code, repair, and oracle prompts.
  It also stores canonical naive decomposition plans, such as the LayerNorm
  seven-kernel baseline.
- `builder/candidate_verifier.py`: verifies generated backward candidates against
  PyTorch autograd and provides syntax/verification command helpers.
- `builder/llm_client.py`: minimal OpenAI-compatible API wrapper.
- `builder/synthesize.py`: orchestrates Stage 1 synthesis: decomposition
  generation, code generation, verification, and repair.

## Common Commands

```bash
python -m benchmark.triton_backward_bench_common.builder_cli emit-spec benchmark/triton_layernorm_backward_bench
python -m benchmark.triton_backward_bench_common.builder_cli emit-prompts benchmark/triton_layernorm_backward_bench
python -m benchmark.triton_backward_bench_common.builder_cli verify benchmark/triton_layernorm_backward_bench
```

Run API-based Stage 1 synthesis on a GPU node:

```bash
python -m benchmark.triton_backward_bench_common.builder_cli synthesize \
  benchmark/triton_layernorm_backward_bench \
  --api-base https://api.openai.com/v1 \
  --model gpt-5.5 \
  --max-attempts 3 \
  --output-dir stage1_candidates
```
