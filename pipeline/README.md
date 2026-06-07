# AtenIR Agent Pipelines

This directory contains the current AtenIR-to-Triton synthesis pipelines.

- Pipeline A: `fusion_agent`, direct AtenIR graph fusion.
- Pipeline B: `primitive_atenir_lowering_agent`, per-op AtenIR lowering to individual Triton kernels.
- Pipeline C: `kernel_fusion_agent`, kernel-aware fusion using verified per-op lowering context.

Use the top-level modules for command-line entry points:

```bash
python -m pipeline.run_fusion_agent --help
python -m pipeline.run_lowering_agent --help
python -m pipeline.run_kernel_fusion_agent --help
python -m pipeline.run_layernorm_pipeline_comparison --help
```

Shared LLM and failure-taxonomy helpers live in `pipeline/shared/`.
