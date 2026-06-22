To run this pipeline:

```python
python -m pipeline.run_handwritten_dispatch \
    --forward atenir._examples:layernorm \
    --example-input "[(8,64) f32, (64) f32, (64) f32]" \
    --output-dir ~/tmp/D_layernorm \
    --dtype float32 --dtype float16
```
