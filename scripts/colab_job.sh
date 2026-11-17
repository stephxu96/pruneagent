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

cleanup() {  # retry, so an offline moment doesn't leave the VM running (and using compute units)
  echo "[job] stopping session $session" >&2
  for _ in 1 2 3 4 5 6 7 8 9 10; do colab stop -s "$session" >&2 && return; sleep 30; done
  echo "[job] could not stop $session; stop it by hand: colab stop -s $session" >&2
}
trap cleanup EXIT

echo "[job] creating $gpu session $session" >&2
colab new -s "$session" --gpu "$gpu" >&2 || { echo "[job] could not create session" >&2; exit 2; }

colab exec -s "$session" --timeout 900 -f "$script" >&2
rc=$?
if [ $rc -ne 0 ]; then
  echo "[job] exec ended with $rc; checking whether the job is still running on the VM" >&2
  # Wait while the job is BUSY. If Colab can't be reached (e.g. the Mac is offline), keep waiting:
  # the job carries on on the VM, so a failed status check must not be taken as "finished".
  for _ in $(seq 1 180); do
    if st=$(colab status -s "$session" 2>&1); then
      if ! grep -qi "busy" <<<"$st"; then break; fi
    else
      echo "[job] Colab not reachable; still waiting" >&2
    fi
    sleep 30
  done
  echo "$st" >&2
fi

mkdir -p "$(dirname "$local_result")"
for attempt in 1 2 3 4 5 6; do  # retry the download for ~5 min in case the connection is still flaky
  if colab download -s "$session" "$remote_result" "$local_result" >&2; then
    echo "[job] result saved to $local_result" >&2
    exit 0
  fi
  echo "[job] download attempt $attempt failed; retrying in 60 s" >&2
  sleep 60
done
echo "[job] no result file on the VM; last log entries:" >&2
colab log -s "$session" -n 30 >&2 || true
exit 1
