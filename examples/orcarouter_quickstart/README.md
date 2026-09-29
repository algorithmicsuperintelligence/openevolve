# OrcaRouter Quickstart

Use [OrcaRouter](https://www.orcarouter.ai) as the LLM backend for OpenEvolve.
OrcaRouter is an OpenAI-compatible AI gateway that routes many providers behind
one endpoint, so a single credential reaches every model in the catalog.

## Setup

Install OpenEvolve and choose **one** of the two credential entries. They are
independent — use whichever suits your machine, and switch at any time.

```bash
pip install openevolve
```

### Option 1 — existing API key (`orcarouter`)

Create a key at <https://www.orcarouter.ai/console/authorized-apps>, then either
export it or store it once:

```bash
export ORCAROUTER_API_KEY="sk-orca-…"       # picked up automatically
python openevolve-run.py connect orcarouter  # or store it for later runs
```

### Option 2 — account sign-in (`orcarouter_oauth`)

OAuth 2.0 + PKCE. The browser flow mints an ordinary OrcaRouter API key that
belongs to your account; no client secret and no pre-registered redirect URI are
involved.

```bash
python openevolve-run.py connect orcarouter_oauth
```

A loopback listener receives the redirect and your browser opens automatically.
On a machine that cannot receive a redirect (SSH session, container), ask for the
out-of-band flow instead and paste the displayed code back:

```bash
python openevolve-run.py connect orcarouter_oauth --oob
```

The issued key is durable. It is reused until you revoke it at
<https://www.orcarouter.ai/console/authorized-apps>; a `401` from the relay marks
that credential for re-authentication rather than retrying it, because there is
no refresh grant.

## Inspect the catalog

Model ids come from the live catalog (`GET https://api.orcarouter.ai/v1/models`),
filtered for the entry point that will use them. Never type a model name by hand:

```bash
python openevolve-run.py models
python openevolve-run.py models --capability embedding
python openevolve-run.py models --capability chat --modality image
python openevolve-run.py status
```

## Run

```bash
python openevolve-run.py \
  examples/orcarouter_quickstart/initial_program.py \
  examples/orcarouter_quickstart/evaluator.py \
  --config examples/orcarouter_quickstart/config.yaml \
  --iterations 50
```

Or drive the provider entirely from the command line:

```bash
python openevolve-run.py \
  examples/orcarouter_quickstart/initial_program.py \
  examples/orcarouter_quickstart/evaluator.py \
  --provider orcarouter \
  --model deepseek/deepseek-v4-pro \
  --iterations 50
```

## Key config options

| Field | Description | Default |
|-------|-------------|---------|
| `provider` | `"orcarouter"` (API key) or `"orcarouter_oauth"` (sign-in) | `"openai"` |
| `api_base` | Inference base URL. Unset means `https://api.orcarouter.ai/v1` | `null` |
| `api_key` | `sk-orca-…` key, or `${ORCAROUTER_API_KEY}` | `null` |
| `orcarouter_oob` | Use the out-of-band code flow instead of the loopback redirect | `false` |
| `name` | A catalog model id, e.g. `deepseek/deepseek-v4-pro` | — |
| `reasoning_effort` | Sent when the model declares a reasoning ladder | `null` |

## Settings UI

`scripts/visualizer.py` exposes an OrcaRouter page where both credential choices
sit side by side and the model selector is populated from the live catalog:

```bash
python scripts/visualizer.py --path examples/orcarouter_quickstart
# http://127.0.0.1:8080/orcarouter
```

The key stays in the local server process; the browser only ever receives a
redacted form of it.
