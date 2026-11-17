#!/usr/bin/env bash
# One speed-measurement session (task 1.6): device test, then the unpruned model and the mixed plan, measured
# twice each in separate processes (tags a and b) to check run-to-run variation.
#
#   scripts/mac_gpu.sh scripts/speed_session.sh m4 mps [extra speed options]     # Mac: hold the GPU lock throughout
#   scripts/speed_session.sh t4 cuda                                              # Colab T4 (scripts/colab_speed.py)
#
# Builds checkpoints/pruned/mixed_check from plans/mixed_check.json first if it is missing
# (set SCORES=<file> to reuse cached scores).
set -euo pipefail
name=$1 device=$2
shift 2
model=checkpoints/pruned/mixed_check
if [[ ! -f $model/plan.json ]]; then
  uv run pruneagent prune --plan plans/mixed_check.json --out "$model" --windows 1 --device "$device" \
    --results results/speed/prune_log ${SCORES:+--scores "$SCORES"}
fi
uv run pruneagent device-test --name "$name" --device "$device"
for tag in a b; do
  uv run pruneagent speed --device "$device" --tag "$tag" "$@"
  uv run pruneagent speed --path "$model" --device "$device" --tag "$tag" "$@"
done
