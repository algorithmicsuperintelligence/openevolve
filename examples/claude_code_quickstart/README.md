# Claude Code CLI Quickstart

This example shows how to use the [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) as the LLM backend for OpenEvolve. No API keys are needed — authentication uses the CLI's existing OAuth session.

## Prerequisites

1. **Install Claude Code CLI:**
   ```bash
   npm install -g @anthropic-ai/claude-code
   ```

2. **Authenticate:**
   ```bash
   claude login
   ```

3. **Install OpenEvolve:**
   ```bash
   pip install openevolve
   ```

## Run

```bash
python openevolve-run.py \
  examples/claude_code_quickstart/initial_program.py \
  examples/claude_code_quickstart/evaluator.py \
  --config examples/claude_code_quickstart/config.yaml \
  --iterations 50
```

## How It Works

The `config.yaml` sets `provider: "claude_code"` which routes all LLM calls through the `claude -p` subprocess instead of the OpenAI-compatible API. The CLI handles authentication, model selection, and billing.

### Key Config Options

| Field | Description | Default |
|-------|-------------|---------|
| `provider` | Set to `"claude_code"` to use the CLI backend | `"openai"` |
| `name` | Claude model name (`sonnet`, `haiku`, `opus`) | `"sonnet"` |
| `max_budget_usd` | Per-call spending cap in USD | `1.0` |
| `timeout` | CLI timeout in seconds | `300` |
| `retries` | Number of retry attempts on failure | `3` |
| `retry_delay` | Seconds between retries | `5` |

### Ensemble Example

You can mix Claude models in an ensemble, just like with OpenAI models:

```yaml
llm:
  provider: "claude_code"
  models:
    - name: "sonnet"
      weight: 0.8
      max_tokens: 16000
    - name: "haiku"
      weight: 0.2
      max_tokens: 8000
```

### Programmatic Usage

You can also inject the Claude Code backend at runtime without modifying config files:

```python
from openevolve.llm.claude_code import init_claude_code_client

for model_cfg in config.llm.models:
    model_cfg.init_client = init_claude_code_client
```
