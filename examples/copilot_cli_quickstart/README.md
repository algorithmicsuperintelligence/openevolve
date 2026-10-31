# GitHub Copilot CLI Quickstart

This example shows how to use the [GitHub Copilot CLI](https://docs.github.com/copilot/how-tos/copilot-cli) as the LLM backend for OpenEvolve. No API keys are needed — authentication uses your GitHub Copilot subscription.

## Prerequisites

1. **Install GitHub Copilot CLI:**
   ```bash
   npm install -g @github/copilot
   ```

2. **Authenticate:**
   ```bash
   copilot login
   ```

   For headless runs you can instead export a fine-grained personal access token with the "Copilot Requests" permission as `COPILOT_GITHUB_TOKEN`.

3. **Install OpenEvolve:**
   ```bash
   pip install openevolve
   ```

## Run

```bash
python openevolve-run.py \
  examples/copilot_cli_quickstart/initial_program.py \
  examples/copilot_cli_quickstart/evaluator.py \
  --config examples/copilot_cli_quickstart/config.yaml \
  --iterations 50
```

## How It Works

The `config.yaml` sets `provider: "copilot_cli"` which routes all LLM calls through the `copilot -p` subprocess instead of the OpenAI-compatible API. The CLI handles authentication, model selection, and billing.

Because the CLI has no `--system-prompt` flag, the system message is prepended to the prompt. Each call runs with `--silent --no-color --no-ask-user --no-custom-instructions --disable-builtin-mcps` so that only the model response reaches stdout and prompts stay reproducible across machines.

### Key Config Options

| Field | Description | Default |
|-------|-------------|---------|
| `provider` | Set to `"copilot_cli"` to use the CLI backend | `"openai"` |
| `name` | Model passed to `--model`; use `"auto"` to let Copilot choose | `"auto"` |
| `reasoning_effort` | One of `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` | unset |
| `max_ai_credits` | Per-call AI credit budget | unset |
| `allow_all_tools` | Pre-approve the agent's tools | `false` |
| `timeout` | CLI timeout in seconds | `300` |
| `retries` | Number of retry attempts on failure | `3` |
| `retry_delay` | Seconds between retries | `5` |

`temperature`, `top_p` and `max_tokens` have no equivalent CLI flag and are ignored by this backend.

Evolution only needs text generation, so tools are left unapproved by default. Set `allow_all_tools: true` only if you want the agent to read and write files while generating.

### Ensemble Example

A single provider covers every model your subscription exposes, so an ensemble can mix vendors:

```yaml
llm:
  provider: "copilot_cli"
  models:
    - name: "claude-sonnet-4.6"
      weight: 0.5
    - name: "gpt-5.4"
      weight: 0.3
    - name: "gemini-3.1-pro-preview"
      weight: 0.2
```

### Programmatic Usage

You can also inject the Copilot CLI backend at runtime without modifying config files:

```python
from openevolve.llm.copilot_cli import init_copilot_cli_client

for model_cfg in config.llm.models:
    model_cfg.init_client = init_copilot_cli_client
```
