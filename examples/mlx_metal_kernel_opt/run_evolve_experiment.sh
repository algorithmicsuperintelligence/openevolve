#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
run_evolve_experiment.sh

Run OpenEvolve for the mlx_metal_kernel_opt example with an isolated output dir
and an isolated database path (so multiple runs don't overwrite each other).

Defaults:
  - config:    ./config.yaml
  - initial:   ./initial_program.py
  - evaluator: ./evaluator.py
  - output:    ./openevolve_output_<timestamp>/
  - db_path:   <output>/qwen3_metal_kernel_evolution

Required env:
  - OPENAI_API_KEY must be set (Gemini OpenAI-compatible API key in this example).
    Alternatively, set GEMINI_API_KEY and the script will map it to OPENAI_API_KEY.

Usage:
  bash run_evolve_experiment.sh [options]

Options:
  --run-name NAME        Output dir name (default: openevolve_output_<timestamp>)
  --output-base DIR      Base directory to create the run directory in (default: example dir)
  --config PATH          Path to config YAML (default: ./config.yaml)
  --python PATH          Python interpreter to use (default: python). The script runs it with -u (unbuffered).
  --iterations N         Override max iterations (passes -i to openevolve CLI)
  --target-score S       Override target score (passes -t)
  --log-level LEVEL      Override log level (passes -l)
  --checkpoint PATH      Resume from checkpoint directory (passes --checkpoint)
  --resume               Auto-resume from the latest run's latest checkpoint
  --api-base URL         Override LLM api_base (passes --api-base)
  --primary-model NAME   Override primary model (passes --primary-model)
  --secondary-model NAME Override secondary model (passes --secondary-model)
  --foreground           Run in foreground (default: background + write run.log)
  --dry-run              Print what would run, but do not execute
  -h, --help             Show this help

Examples:
  export OPENAI_API_KEY="..."
  bash run_evolve_experiment.sh                    # Start new run
  bash run_evolve_experiment.sh --resume           # Resume latest run
  bash run_evolve_experiment.sh --iterations 5 --run-name trial_5iter
USAGE
}

# Force unbuffered Python output for reliable logging
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_NAME=""
OUTPUT_BASE="$SCRIPT_DIR"
CONFIG_PATH="$SCRIPT_DIR/config.yaml"
PYTHON_BIN="python"

ITERATIONS=""
TARGET_SCORE=""
LOG_LEVEL=""
CHECKPOINT=""
API_BASE=""
PRIMARY_MODEL=""
SECONDARY_MODEL=""

FOREGROUND=0
DRY_RUN=0
RESUME=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name) RUN_NAME="${2:-}"; shift 2 ;;
    --output-base) OUTPUT_BASE="${2:-}"; shift 2 ;;
    --config) CONFIG_PATH="${2:-}"; shift 2 ;;
    --python) PYTHON_BIN="${2:-}"; shift 2 ;;
    --iterations) ITERATIONS="${2:-}"; shift 2 ;;
    --target-score) TARGET_SCORE="${2:-}"; shift 2 ;;
    --log-level) LOG_LEVEL="${2:-}"; shift 2 ;;
    --checkpoint) CHECKPOINT="${2:-}"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --api-base) API_BASE="${2:-}"; shift 2 ;;
    --primary-model) PRIMARY_MODEL="${2:-}"; shift 2 ;;
    --secondary-model) SECONDARY_MODEL="${2:-}"; shift 2 ;;
    --foreground) FOREGROUND=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${OPENAI_API_KEY:-}" && -n "${GEMINI_API_KEY:-}" ]]; then
  # Convenience: allow users to keep the key in GEMINI_API_KEY while OpenEvolve expects OPENAI_API_KEY.
  export OPENAI_API_KEY="${GEMINI_API_KEY}"
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set. Export OPENAI_API_KEY (or GEMINI_API_KEY) before running." >&2
  exit 1
fi

# Handle --resume: find the latest run directory and its latest checkpoint
if [[ "$RESUME" -eq 1 ]]; then
  echo "[run_evolve_experiment] --resume: Looking for latest run to continue..."
  
  # Find the latest openevolve_output_* directory
  LATEST_RUN_DIR=$(find "$OUTPUT_BASE" -maxdepth 1 -type d -name "openevolve_output_*" 2>/dev/null | sort -r | head -n 1)
  
  if [[ -z "$LATEST_RUN_DIR" ]]; then
    echo "ERROR: No previous run found in $OUTPUT_BASE. Cannot resume." >&2
    echo "       Start a new run without --resume first." >&2
    exit 1
  fi
  
  # Extract run name from path
  RUN_NAME=$(basename "$LATEST_RUN_DIR")
  echo "[run_evolve_experiment] Found latest run: $RUN_NAME"
  
  # Find the latest checkpoint in that run
  CHECKPOINT_DIR="$LATEST_RUN_DIR/checkpoints"
  if [[ -d "$CHECKPOINT_DIR" ]]; then
    # Sort by the numeric suffix of checkpoint_N (extract number, sort numerically)
    LATEST_CHECKPOINT=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name "checkpoint_*" 2>/dev/null | \
      while read -r p; do echo "$(basename "$p" | sed 's/checkpoint_//')|$p"; done | \
      sort -t'|' -k1 -n | tail -n 1 | cut -d'|' -f2)
    
    if [[ -n "$LATEST_CHECKPOINT" ]]; then
      CHECKPOINT="$LATEST_CHECKPOINT"
      echo "[run_evolve_experiment] Found latest checkpoint: $CHECKPOINT"
    else
      echo "[run_evolve_experiment] No checkpoint found, will continue from database state"
    fi
  else
    echo "[run_evolve_experiment] No checkpoints directory found, will continue from database state"
  fi
fi

if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="openevolve_output_$(date +%Y%m%d_%H%M%S)"
fi

RUN_DIR="$OUTPUT_BASE/$RUN_NAME"
mkdir -p "$RUN_DIR"

INITIAL_PROGRAM="$SCRIPT_DIR/initial_program.py"
EVALUATION_FILE="$SCRIPT_DIR/evaluator.py"

CFG_OUT="$RUN_DIR/config.yaml"

# Write an updated config into the run directory with db_path isolated to this run.
CONFIG_PATH="$CONFIG_PATH" CFG_OUT="$CFG_OUT" RUN_DIR="$RUN_DIR" "$PYTHON_BIN" -u - <<'PY'
import os
import sys

import yaml

cfg_path = os.environ["CONFIG_PATH"]
out_path = os.environ["CFG_OUT"]
run_dir = os.environ["RUN_DIR"]

with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

cfg = cfg or {}
cfg.setdefault("database", {})
cfg["database"]["db_path"] = os.path.join(run_dir, "qwen3_metal_kernel_evolution")

with open(out_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"[run_evolve_experiment] Wrote config: {out_path}")
print(f"[run_evolve_experiment] database.db_path: {cfg['database']['db_path']}")
PY

CMD=(
  "$PYTHON_BIN" -u -m openevolve.cli
  "$INITIAL_PROGRAM"
  "$EVALUATION_FILE"
  -c "$CFG_OUT"
  -o "$RUN_DIR"
)

if [[ -n "$ITERATIONS" ]]; then CMD+=(-i "$ITERATIONS"); fi
if [[ -n "$TARGET_SCORE" ]]; then CMD+=(-t "$TARGET_SCORE"); fi
if [[ -n "$LOG_LEVEL" ]]; then CMD+=(-l "$LOG_LEVEL"); fi
if [[ -n "$CHECKPOINT" ]]; then CMD+=(--checkpoint "$CHECKPOINT"); fi
if [[ -n "$API_BASE" ]]; then CMD+=(--api-base "$API_BASE"); fi
if [[ -n "$PRIMARY_MODEL" ]]; then CMD+=(--primary-model "$PRIMARY_MODEL"); fi
if [[ -n "$SECONDARY_MODEL" ]]; then CMD+=(--secondary-model "$SECONDARY_MODEL"); fi

echo "[run_evolve_experiment] Run dir: $RUN_DIR"
echo "[run_evolve_experiment] Command:"
printf "  %q" "${CMD[@]}"
echo

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

LOG_FILE="$RUN_DIR/run.log"

# Truncate log file to ensure clean start (especially important for --resume)
: > "$LOG_FILE"

if [[ "$FOREGROUND" -eq 1 ]]; then
  # Stream to console and persist logs with line buffering.
  if command -v stdbuf &>/dev/null; then
    stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
  else
    "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
  fi
else
  # Run in background with line-buffered output for reliable log ordering.
  if command -v stdbuf &>/dev/null; then
    nohup stdbuf -oL -eL "${CMD[@]}" > "$LOG_FILE" 2>&1 &
  else
    nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &
  fi
  echo "[run_evolve_experiment] Started PID: $!"
  echo "[run_evolve_experiment] Log: $LOG_FILE"
  echo "[run_evolve_experiment] Tail: tail -f \"$LOG_FILE\""
fi


