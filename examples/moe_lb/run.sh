#!/bin/bash

WORKLOAD_DIR="examples/moe_lb"

python ./openevolve-run.py \
    $WORKLOAD_DIR/initial_program.py \
    $WORKLOAD_DIR/evaluator.py \
    -c $WORKLOAD_DIR/config.yaml \
    -o out/moe_lb
