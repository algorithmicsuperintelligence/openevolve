#!/bin/bash

TASK_ID=${1:-0} # Task number to solve (default: 0)
shift  # Remove first argument
TASK_FILE="evaluation" # Options: training, evaluation, test (default: evaluation)

export OPENAI_API_KEY="your-gemini-api-key"
export ARC_TASK_FILE=$TASK_FILE  
export TASK_NUM=$TASK_ID
export DATA_ROOT="../../data/arc-prize-2025"

OUTPUT_DIR="outputs/${TASK_FILE}_task_${TASK_ID}"
export OUTS_DIR=$OUTPUT_DIR

python generate_config.py
python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml --output $OUTPUT_DIR "$@"
python post_evolution_eval.py
