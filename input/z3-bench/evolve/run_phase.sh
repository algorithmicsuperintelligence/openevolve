#!/bin/bash
# Run one phase of Z3 parameter evolution.
# Usage:
#   ./run_phase.sh {1|2|3|4} [extra openevolve-run.py flags...]
#   ./run_phase.sh {1|2|3} --extract-only        # skip evolution; aggregate
#                                                # results from existing
#                                                # openevolve_output/checkpoints/
#
# Iteration count comes from config.yaml (max_iterations). Override via
# `--iterations <N>` extra flag if needed.
# OpenEvolve outputs land in <phase_dir>/openevolve_output/.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 {1|2|3|4} [--extract-only] [extra flags]" >&2
    exit 2
fi

PHASE="$1"
shift

EXTRACT_ONLY=0
PASSTHROUGH=()
for arg in "$@"; do
    case "$arg" in
        --extract-only) EXTRACT_ONLY=1 ;;
        *) PASSTHROUGH+=("$arg") ;;
    esac
done
set -- "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}"

case "$PHASE" in
    1) DIR="phase1_opt_sls" ;;
    2) DIR="phase2_sat"     ;;
    3) DIR="phase3_smt"     ;;
    4) DIR="phase4_unified" ;;
    *) echo "phase must be 1, 2, 3, or 4 (got $PHASE)" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../../.." && pwd)"
RUNNER="$REPO_ROOT/openevolve-run.py"

# --extract-only: skip evolution + setup, jump straight to aggregation.
if [ "$EXTRACT_ONLY" = "1" ]; then
    if [ "$PHASE" -eq 4 ]; then
        echo "--extract-only not supported for phase 4 (no extract step)" >&2
        exit 2
    fi
    echo "[run_phase] --extract-only: aggregating from checkpoints (phase $PHASE)..."
    python "$ROOT/extract_best.py" "$PHASE" --from-checkpoints
    echo "[run_phase] extract-only done. next: $0 $((PHASE + 1))"
    exit 0
fi

if [ ! -f "$RUNNER" ]; then
    echo "openevolve-run.py not found at $RUNNER" >&2
    exit 1
fi

if [ ! -f "$ROOT/shared/stage1_sample.json" ] || [ ! -f "$ROOT/shared/stage2_sample.json" ]; then
    echo "sample json missing — running build_samples.py first..."
    python "$ROOT/build_samples.py"
fi

if [ "${SKIP_REBASELINE:-0}" = "1" ] && [ -f "$ROOT/shared/local_baseline.json" ]; then
    echo "SKIP_REBASELINE=1 set — reusing existing local_baseline.json"
elif [ "${SKIP_REBASELINE:-0}" = "1" ]; then
    echo "SKIP_REBASELINE=1 set but local_baseline.json missing — running rebaseline_local.py..."
    python "$ROOT/rebaseline_local.py" || \
        echo "warning: rebaseline_local.py finished with mismatches; evaluator will fall back to raw_ms for those."
else
    echo "running rebaseline_local.py (~5 min, 20 problems)... set SKIP_REBASELINE=1 to skip"
    python "$ROOT/rebaseline_local.py" || \
        echo "warning: rebaseline_local.py finished with mismatches; evaluator will fall back to raw_ms for those."
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
echo "[run_phase] phase=$PHASE dir=$DIR cwd=$(pwd) (iterations from config.yaml)"

python "$RUNNER" \
    initial_program.py \
    "$ROOT/shared/evaluator.py" \
    --config "$ROOT/config.yaml" \
    "$@"

echo "[run_phase] phase $PHASE finished."

if [ "$PHASE" -lt 4 ]; then
    echo "[run_phase] extracting best overrides..."
    python "$ROOT/extract_best.py" "$PHASE"
    echo "[run_phase] next: $0 $((PHASE + 1))"
fi
