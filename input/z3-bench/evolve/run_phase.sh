#!/bin/bash
# Run one phase of Z3 parameter evolution.
# Usage: ./run_phase.sh {1|2|3|4} [extra openevolve-run.py flags...]
#
# Phase iterations are set per-phase below. Override via --iterations <N>.
# OpenEvolve outputs land in <phase_dir>/openevolve_output/.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 {1|2|3|4} [extra flags]" >&2
    exit 2
fi

PHASE="$1"
shift

case "$PHASE" in
    1) DIR="phase1_opt_sls"; ITERS=80  ;;
    2) DIR="phase2_sat";     ITERS=150 ;;
    3) DIR="phase3_smt";     ITERS=120 ;;
    4) DIR="phase4_unified"; ITERS=60  ;;
    *) echo "phase must be 1, 2, 3, or 4 (got $PHASE)" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../../.." && pwd)"
RUNNER="$REPO_ROOT/openevolve-run.py"

if [ ! -f "$RUNNER" ]; then
    echo "openevolve-run.py not found at $RUNNER" >&2
    exit 1
fi

if [ ! -f "$ROOT/shared/stage1_sample.json" ]; then
    echo "stage1_sample.json missing — running build_stage1_sample.py first..."
    python "$ROOT/build_stage1_sample.py"
fi

if ! command -v z3 >/dev/null 2>&1; then
    echo "warning: z3 binary not on PATH. install: apt-get install -y z3  or  pip install z3-solver" >&2
fi

if [ "$PHASE" -eq 4 ]; then
    if [ ! -f "$ROOT/shared/phase1_best.json" ] \
        || [ ! -f "$ROOT/shared/phase2_best.json" ] \
        || [ ! -f "$ROOT/shared/phase3_best.json" ]; then
        echo "phase 4 requires phase 1/2/3 best.json files in shared/." >&2
        echo "run phases 1-3 first." >&2
        exit 1
    fi
    echo "[run_phase] materializing phase4 EVOLVE-BLOCK..."
    python "$ROOT/prepare_phase4.py"
fi

cd "$ROOT/$DIR"
echo "[run_phase] phase=$PHASE dir=$DIR iters=$ITERS cwd=$(pwd)"

python "$RUNNER" \
    initial_program.py \
    "$ROOT/shared/evaluator.py" \
    --config "$ROOT/config.yaml" \
    --iterations "$ITERS" \
    "$@"

echo "[run_phase] phase $PHASE finished."

if [ "$PHASE" -lt 4 ]; then
    echo "[run_phase] extracting best overrides..."
    python "$ROOT/extract_best.py" "$PHASE"
    echo "[run_phase] next: $0 $((PHASE + 1))"
fi
