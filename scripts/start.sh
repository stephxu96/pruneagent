#!/usr/bin/env bash
# Everything in one go: install what's needed, build the current release and the interface, run the web app and the
# workflow service in the background (started at login, restarted if they stop), then open the app.
#   scripts/start.sh            # first time, and after every update (safe to run again)
#   scripts/start.sh --stop     # stop both and take them out of the login items
#   scripts/start.sh --no-open  # don't open the browser
set -euo pipefail
cd "$(dirname "$0")/.."
url="http://127.0.0.1:8125"

if [[ "${1:-}" == "--stop" ]]; then
  scripts/install_services.sh --remove
  echo "Stopped. Work in progress pauses at its current step; scripts/start.sh continues it."
  exit 0
fi

echo "== Packages"
uv sync --extra web --extra report --quiet
[[ -d web/node_modules ]] || (cd web && npm install --silent)

echo "== Release and interface"
scripts/deploy.sh | grep -v "isn't installed as a background service" || true

echo "== Background services"
uid="$(id -u)"
if launchctl print "gui/$uid/com.pruneagent.web" >/dev/null 2>&1 && launchctl print "gui/$uid/com.pruneagent.worker" >/dev/null 2>&1; then
  echo "Already installed (restarted by the deploy above)"
else
  scripts/install_services.sh
fi

echo "== Checking"
for _ in $(seq 1 60); do
  status="$(curl -sf "$url/api/service" 2>/dev/null || true)"
  [[ "$status" == *'"running":true'* ]] && break
  sleep 1
done
if [[ "$status" != *'"running":true'* ]]; then
  echo "Something didn't start. Logs: runs/service_logs/web.log and runs/service_logs/worker.log" >&2
  exit 1
fi
echo "Web app: $url"
echo "Workflow service: running ($status)"
[[ "${1:-}" == "--no-open" ]] || open "$url"
