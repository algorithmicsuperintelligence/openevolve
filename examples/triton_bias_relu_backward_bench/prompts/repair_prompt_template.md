# Repair Prompt Template

The naive Triton backward candidate for `bias_relu` failed verification.

Forward semantic ground truth:

```text
y = relu(x + bias)
```

Required gradients:

```text
dx, dbias
```

Expected oracle:

```text
PyTorch autograd over forward_ref.py
```

Insert verifier feedback below before sending this prompt to an agent:

```text
{VERIFIER_ERROR_REPORT}
```

Repair task:

1. Explain the likely mathematical or indexing error.
2. Modify only the naive Triton backward candidate.
3. Preserve the public API and output order.
4. Keep the implementation simple, readable, and unfused.
5. Do not weaken tests or tolerances.
6. Do not add argument-order compatibility hacks.
