#!/usr/bin/env bash
# Run a Python job on a named Colab GPU session and always release it.
#
#   scripts/colab_job.sh <session> <gpu> <script.py> <local-result.json> [remote-result]
#
# Creates the session, runs the script with `colab exec`. If the connection drops, the kernel keeps
# running on the VM, so this waits until the session is no longer BUSY (up to 90 min), then downloads
# the result file (default /content/result.json). The session is stopped on exit, success or not.
set -uo pipefail
session=$1 gpu=$2 script=$3 local_result=$4 remote_result=${5:-/content/result.json}

cleanup() { echo "[job] stopping session $session" >&2; colab stop -s "$session" >&2 || true; }
trap cleanup EXIT

echo "[job] creating $gpu session $session" >&2
colab new -s "$session" --gpu "$gpu" >&2 || { echo "[job] could not create session" >&2; exit 2; }

colab exec -s "$session" --timeout 900 -f "$script" >&2
rc=$?
if [ $rc -ne 0 ]; then
  echo "[job] exec ended with $rc; checking whether the job is still running on the VM" >&2
  for _ in $(seq 1 180); do
    st=$(colab status -s "$session" 2>&1)
    if ! grep -qi "busy" <<<"$st"; then break; fi
    sleep 30
  done
  echo "$st" >&2
fi

if colab download -s "$session" "$remote_result" "$local_result" >&2; then
  echo "[job] result saved to $local_result" >&2
  exit 0
fi
echo "[job] no result file on the VM; last log entries:" >&2
colab log -s "$session" -n 30 >&2 || true
exit 1
