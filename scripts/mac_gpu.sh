#!/usr/bin/env bash
# Run a command with exclusive use of the Mac GPU; other callers wait their turn.
#   scripts/mac_gpu.sh uv run pruneagent flap --ratio 0.2
# The lock is a directory (atomic mkdir); a lock left by a dead process is cleared.
set -euo pipefail
lock="${TMPDIR:-/tmp}/pruneagent-mac-gpu.lock"
waited=0
until mkdir "$lock" 2>/dev/null; do
  holder=$(cat "$lock/pid" 2>/dev/null || true)
  if [[ -n "$holder" ]] && ! kill -0 "$holder" 2>/dev/null; then
    rm -rf "$lock"; continue
  fi
  if (( waited % 60 == 0 )); then
    echo "[mac_gpu] waiting for Mac GPU (held by pid ${holder:-?}: $(cat "$lock/cmd" 2>/dev/null || echo '?'))" >&2
  fi
  sleep 5; waited=$((waited + 5))
done
echo $$ > "$lock/pid"
echo "$*" > "$lock/cmd"
trap 'rm -rf "$lock"' EXIT
"$@"
