#!/usr/bin/env bash
# Estimate-vs-measured speed check (task 1.7): build every plans/speed_check_*.json and measure it with the speed
# meter. Each measurement is written to results/speed_check/measured/ as soon as it is taken; plans already
# measured on this device are skipped, so a rerun picks up where it stopped.
#
#   scripts/mac_gpu.sh scripts/speed_check_session.sh mps     # Mac: hold the GPU lock throughout
#   scripts/speed_check_session.sh cuda                       # Colab T4 (scripts/colab_speed_check.py)
#
# Set SCORES=<file> to reuse cached pruning scores and DATA=<dir> for a data cache outside ./data_cache.
set -euo pipefail
device=$1
shift
out=results/speed_check/measured
for plan in plans/speed_check_*.json; do
  name=$(basename "$plan" .json)
  if ls "$out"/*__"${name}"__"${device}"__*.json >/dev/null 2>&1; then
    echo "[speed_check] $name already measured on $device" >&2
    continue
  fi
  model=checkpoints/pruned/$name
  if [[ ! -f $model/plan.json ]]; then
    uv run pruneagent prune --plan "$plan" --out "$model" --windows 1 --device "$device" \
      --results results/speed_check/prune_log ${SCORES:+--scores "$SCORES"} ${DATA:+--data-dir "$DATA"}
  fi
  uv run pruneagent speed --path "$model" --device "$device" --results "$out" ${DATA:+--data-dir "$DATA"} "$@"
done
