#!/bin/bash
# Single entry for all bench phase runs. Reads `bench:` section of
# <bench>/evolve/config.yaml (via _lib/load_bench_config.py), then drives the
# standard flow:
#   parse flags → rebaseline → solver check → openevolve-run → extract_best.
#
# Usage:
#   ./input/run_phase.sh <bench> [<phase>] [--pin N-M] [--extract-only]
#                                          [--iterations <N>] [extra flags]
#
#   <phase> omitted  → run ALL phases sequentially (1..N).
#   <phase> numeric  → run that single phase.
#
# Examples:
#   ./input/run_phase.sh cpsat-bench           # run all phases (1..4)
#   ./input/run_phase.sh cpsat-bench --pin 2-7 # all phases, pinned
#   ./input/run_phase.sh cpsat-bench 1 --pin 2-7
#   ./input/run_phase.sh z3-bench 4
#   ./input/run_phase.sh cpsat-bench 2 --extract-only
#
# Bench config (in <bench>/evolve/config.yaml under `bench:` key — see
# _lib/load_bench_config.py for schema). Loader exports these into env:
#   PHASE_DIRS             space-separated phase dir names, in order
#                          (last entry = "unified"/final phase, no extract step)
#   PHASE_ITERS            space-separated iter counts per phase (omitted =>
#                          use config.yaml max_iterations only)
#   UNIFIED_PREPARE_SCRIPT (optional) script name run before the last phase
#   SOLVER_CHECK_CMD       (optional) shell command returning 0 if solver works
#   SOLVER_INSTALL_HINT    (optional) message on solver check fail
#   REBASELINE_SCRIPT      (optional, default rebaseline_local.py)
#   EXTRACT_BEST_SCRIPT    (optional, default extract_best.py)
#   BUILD_SAMPLES_SCRIPT   (optional, default build_samples.py)
#
# Env overrides (any bench):
#   SKIP_REBASELINE=1                 reuse existing local_baseline.json
#   OPENEVOLVE_PARALLEL_SOLVERS=N     concurrent solver subprocesses
#   OPENEVOLVE_CORE_RANGE=N-M         taskset core range (also set via --pin)

set -euo pipefail

usage() {
    echo "usage: $(basename "$0") <bench> [<phase>] [--pin N-M] [--extract-only] [--iterations N] [extra flags]" >&2
    echo "       omit <phase> to run all phases sequentially" >&2
    echo "       <bench> = dir name under input/ (e.g. cpsat-bench, z3-bench)" >&2
}

if [ $# -lt 1 ]; then
    usage
    exit 2
fi

BENCH="$1"; shift

# Optional <phase>: numeric next arg.
PHASE=""
if [ $# -ge 1 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    PHASE="$1"; shift
fi

INPUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$INPUT_DIR/$BENCH/evolve"
CONFIG_YAML="$ROOT/config.yaml"

if [ ! -d "$ROOT" ]; then
    echo "bench evolve dir not found: $ROOT" >&2
    exit 2
fi
if [ ! -f "$CONFIG_YAML" ]; then
    echo "missing $CONFIG_YAML" >&2
    exit 2
fi

# Load `bench:` section from config.yaml → exports PHASE_DIRS, PHASE_ITERS,
# UNIFIED_PREPARE_SCRIPT, SOLVER_CHECK_CMD, ...
_BENCH_EXPORTS="$(python "$INPUT_DIR/_lib/load_bench_config.py" "$CONFIG_YAML")"
eval "$_BENCH_EXPORTS"

: "${PHASE_DIRS:?bench.phases missing in $CONFIG_YAML}"
REBASELINE_SCRIPT="${REBASELINE_SCRIPT:-rebaseline_local.py}"
EXTRACT_BEST_SCRIPT="${EXTRACT_BEST_SCRIPT:-extract_best.py}"
BUILD_SAMPLES_SCRIPT="${BUILD_SAMPLES_SCRIPT:-build_samples.py}"

read -r -a _DIRS <<< "$PHASE_DIRS"
read -r -a _ITERS <<< "${PHASE_ITERS:-}"
N_PHASES=${#_DIRS[@]}
LAST_PHASE=$N_PHASES
PHASE_RANGE_STR="1..$N_PHASES"

# Parse remaining flags
EXTRACT_ONLY=0
PIN_RANGE=""
PASSTHROUGH=()
while [ $# -gt 0 ]; do
    case "$1" in
        --extract-only)
            EXTRACT_ONLY=1
            shift
            ;;
        --pin)
            PIN_RANGE="$2"
            shift 2
            ;;
        --pin=*)
            PIN_RANGE="${1#--pin=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            PASSTHROUGH+=("$1")
            shift
            ;;
    esac
done
set -- "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}"

# Decide phase list
if [ -z "$PHASE" ]; then
    if [ "$EXTRACT_ONLY" = "1" ]; then
        echo "--extract-only requires explicit <phase> (refuses to bulk-aggregate all)" >&2
        exit 2
    fi
    PHASES_TO_RUN=()
    for ((i = 1; i <= N_PHASES; i++)); do
        PHASES_TO_RUN+=("$i")
    done
    RUN_ALL=1
else
    if [ "$PHASE" -lt 1 ] || [ "$PHASE" -gt "$N_PHASES" ]; then
        echo "phase must be in {$PHASE_RANGE_STR} (got: $PHASE)" >&2
        exit 2
    fi
    PHASES_TO_RUN=("$PHASE")
    RUN_ALL=0
fi

if [ -n "$PIN_RANGE" ]; then
    if ! [[ "$PIN_RANGE" =~ ^[0-9]+(-[0-9]+)?$ ]]; then
        echo "--pin expects N or N-M (got: $PIN_RANGE)" >&2
        exit 2
    fi
    export OPENEVOLVE_CORE_RANGE="$PIN_RANGE"
    echo "[run_phase] OPENEVOLVE_CORE_RANGE=$PIN_RANGE"
fi

REPO_ROOT="$(cd "$INPUT_DIR/.." && pwd)"
RUNNER="$REPO_ROOT/openevolve-run.py"

# ============ pre-flight (once, even in run-all mode) ============

if [ "$EXTRACT_ONLY" != "1" ]; then
    if [ ! -f "$RUNNER" ]; then
        echo "openevolve-run.py not found at $RUNNER" >&2
        exit 1
    fi

    if [ ! -f "$ROOT/shared/stage1_sample.json" ] || [ ! -f "$ROOT/shared/stage2_sample.json" ]; then
        echo "[run_phase] sample json missing — running $BUILD_SAMPLES_SCRIPT first..."
        python "$ROOT/$BUILD_SAMPLES_SCRIPT"
    fi

    if [ -f "$ROOT/$REBASELINE_SCRIPT" ]; then
        if [ "${SKIP_REBASELINE:-0}" = "1" ] && [ -f "$ROOT/shared/local_baseline.json" ]; then
            echo "[run_phase] SKIP_REBASELINE=1 — reusing existing local_baseline.json"
        elif [ "${SKIP_REBASELINE:-0}" = "1" ]; then
            echo "[run_phase] SKIP_REBASELINE=1 but local_baseline.json missing — running $REBASELINE_SCRIPT..."
            python "$ROOT/$REBASELINE_SCRIPT" || \
                echo "warning: $REBASELINE_SCRIPT finished with mismatches; evaluator falls back to raw_ms for those."
        else
            echo "[run_phase] running $REBASELINE_SCRIPT (set SKIP_REBASELINE=1 to skip on re-runs)..."
            python "$ROOT/$REBASELINE_SCRIPT" || \
                echo "warning: $REBASELINE_SCRIPT finished with mismatches; evaluator falls back to raw_ms for those."
        fi
    fi

    if [ -n "${SOLVER_CHECK_CMD:-}" ]; then
        if ! eval "$SOLVER_CHECK_CMD" >/dev/null 2>&1; then
            echo "warning: solver check failed. ${SOLVER_INSTALL_HINT:-}" >&2
        fi
    fi

    # Re-using local_baseline across the phase loop is the right thing to do —
    # force-set SKIP_REBASELINE=1 so the inner per-phase logic (if any) doesn't
    # rerun. Currently the inner logic does NOT rerun (we did it above), but
    # set anyway for any subscript that honors it.
    export SKIP_REBASELINE=1
fi

# ============ phase runner (called per phase) ============

run_one_phase() {
    local phase="$1"
    shift
    local dir="${_DIRS[$((phase - 1))]}"
    local iter=""
    if [ "${#_ITERS[@]}" -ge "$phase" ]; then
        iter="${_ITERS[$((phase - 1))]}"
    fi

    if [ "$EXTRACT_ONLY" = "1" ]; then
        if [ "$phase" -eq "$LAST_PHASE" ]; then
            echo "--extract-only not supported for phase $LAST_PHASE (no extract step)" >&2
            return 2
        fi
        echo "[run_phase] --extract-only phase $phase: aggregating from checkpoints..."
        python "$ROOT/$EXTRACT_BEST_SCRIPT" "$phase" --from-checkpoints
        return 0
    fi

    # Unified-prep on the last phase
    if [ "$phase" -eq "$LAST_PHASE" ] && [ -n "${UNIFIED_PREPARE_SCRIPT:-}" ]; then
        local missing=0
        for ((i = 1; i < LAST_PHASE; i++)); do
            if [ ! -f "$ROOT/shared/phase${i}_best.json" ]; then
                echo "phase $LAST_PHASE requires shared/phase${i}_best.json (run phase $i first)" >&2
                missing=1
            fi
        done
        [ "$missing" = "1" ] && return 1
        echo "[run_phase] materializing phase$LAST_PHASE EVOLVE-BLOCK via $UNIFIED_PREPARE_SCRIPT..."
        python "$ROOT/$UNIFIED_PREPARE_SCRIPT"
    fi

    cd "$ROOT/$dir"
    echo "[run_phase] === bench=$BENCH phase=$phase dir=$dir ${iter:+iter=$iter }cwd=$(pwd) ==="

    local iter_flag=()
    if [ -n "$iter" ]; then
        iter_flag=(--iterations "$iter")
    fi

    python "$RUNNER" \
        initial_program.py \
        "$ROOT/shared/evaluator.py" \
        --config "$ROOT/config.yaml" \
        "${iter_flag[@]+"${iter_flag[@]}"}" \
        "$@"

    echo "[run_phase] phase $phase finished."

    if [ "$phase" -lt "$LAST_PHASE" ]; then
        echo "[run_phase] extracting best overrides for phase $phase..."
        python "$ROOT/$EXTRACT_BEST_SCRIPT" "$phase"
    fi
}

# ============ drive ============

for p in "${PHASES_TO_RUN[@]}"; do
    run_one_phase "$p" "$@"
done

if [ "$RUN_ALL" = "1" ]; then
    echo "[run_phase] all $N_PHASES phases completed for $BENCH."
elif [ "${PHASES_TO_RUN[0]}" -lt "$LAST_PHASE" ]; then
    echo "[run_phase] next: $0 $BENCH $((${PHASES_TO_RUN[0]} + 1))"
fi
