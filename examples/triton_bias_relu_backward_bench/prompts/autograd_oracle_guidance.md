# Autograd Oracle Guidance

`backward_ref.py` should be the correctness oracle for `triton_bias_relu_backward_bench`.
It must compute `dx, dbias` by running PyTorch autograd over `forward_ref.py`.

Semantic source:

```text
y = relu(x + bias)
```

Differentiable inputs:

```text
x, bias
```

Required gradients:

```text
dx, dbias
```

Policy:

- Do not use LLM-derived formulas as ground truth.
- Clone inputs before setting `requires_grad_(True)`.
- Call the PyTorch forward reference.
- Call `.backward(dy)` or `torch.autograd.grad(...)`.
- Return gradients in the exact order listed in `required_gradients`.
