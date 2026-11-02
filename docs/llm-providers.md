# LLM Providers

OpenEvolve works with any OpenAI-compatible API, and with the Claude Code, GitHub Copilot and Codex CLIs without an API key.

### OpenAI (Direct)

```bash
export OPENAI_API_KEY="sk-..."
# Uses OpenAI endpoints by default
```

### Google Gemini

```yaml
# config.yaml
llm:
  api_base: "https://generativelanguage.googleapis.com/v1beta/openai/"
  primary_model: "gemini-2.5-pro"
```

```bash
export OPENAI_API_KEY="your-gemini-api-key"
```

### Local Models (Ollama/vLLM)

```yaml
# config.yaml
llm:
  api_base: "http://localhost:11434/v1"  # Ollama
  primary_model: "codellama:7b"
```

### OptiLLM (Advanced)

For maximum flexibility with rate limiting, model routing, and test-time compute:

```bash
# Install OptiLLM
pip install optillm

# Start OptiLLM proxy
optillm --port 8000

# Point OpenEvolve to OptiLLM
export OPENAI_API_KEY="your-actual-key"
```

```yaml
llm:
  api_base: "http://localhost:8000/v1"
  primary_model: "moa&readurls-o3"  # Test-time compute + web access
```

### Claude Code CLI (No API Key)

Use the [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) as the LLM backend — no API keys needed, authentication uses the CLI's OAuth session.

```bash
# Install and authenticate
npm install -g @anthropic-ai/claude-code
claude login
```

```yaml
# config.yaml
llm:
  provider: "claude_code"
  models:
    - name: "sonnet"
      weight: 0.8
      max_tokens: 16000
      max_budget_usd: 1.0
    - name: "haiku"
      weight: 0.2
      max_tokens: 8000
```

See the [Claude Code quickstart example](../examples/claude_code_quickstart/) for a complete walkthrough.

### GitHub Copilot CLI (No API Key)

Use the [GitHub Copilot CLI](https://docs.github.com/copilot/how-tos/copilot-cli) as the LLM backend — no API keys needed, authentication uses your GitHub Copilot subscription.

```bash
# Install and authenticate
npm install -g @github/copilot
copilot login
```

```yaml
# config.yaml
llm:
  provider: "copilot_cli"
  models:
    - name: "claude-sonnet-4.6"
      weight: 0.8
      reasoning_effort: "medium"
    - name: "gpt-5.4"
      weight: 0.2
```

See the [Copilot CLI quickstart example](../examples/copilot_cli_quickstart/) for a complete walkthrough.

### Codex App Server (No API Key)

Use the [Codex CLI's app-server](https://developers.openai.com/codex/app-server/) as the LLM backend — no API keys needed. Authentication uses your ChatGPT (Codex) subscription: the app-server owns the managed browser/device-code login and refreshes the tokens itself. OpenEvolve never reads the CLI's token files, and before each generation it verifies with the app-server that a ChatGPT-plan login is active (it fails with a clear message pointing at `codex login` otherwise). To bill an API key instead, keep the default OpenAI provider — both paths coexist.

```bash
# Install and authenticate
npm install -g @openai/codex
codex login          # browser flow, or: codex login --device-code
```

```yaml
# config.yaml
llm:
  provider: "codex_app_server"
  models:
    - name: "gpt-5.3-codex"
      weight: 0.8
      reasoning_effort: "medium"
    - name: "gpt-5.3-codex-mini"
      weight: 0.2
```

Each call speaks newline-delimited JSON-RPC to a short-lived `codex app-server` process over stdio (unsupported on Windows). Threads run non-interactive: tool approvals are `never` and the sandbox is read-only, so responses are generated without mutating your workspace.

## Cost Estimation

**Cost depends on your LLM provider and iterations:**

- **o3**: ~$0.15-0.60 per iteration (depending on code size)
- **o3-mini**: ~$0.03-0.12 per iteration (more cost-effective)
- **Gemini-2.5-Pro**: ~$0.08-0.30 per iteration
- **Gemini-2.5-Flash**: ~$0.01-0.05 per iteration (fastest and cheapest)
- **Local models**: Nearly free after setup
- **OptiLLM**: Use cheaper models with test-time compute for better results

**Cost-saving tips:**
- Start with fewer iterations (100-200)
- Use o3-mini, Gemini-2.5-Flash or local models for exploration
- Use cascade evaluation to filter bad programs early
- Configure smaller population sizes initially
