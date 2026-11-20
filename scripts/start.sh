#!/usr/bin/env bash
# PruneAgent in one go (docs/APP_DESIGN.md §4.1): the tools natively on this machine (they need its GPU), the app in
# a container. Running it again applies an update; running work is never interrupted.
#   scripts/start.sh            start, or update and restart, both
#   scripts/start.sh --stop     stop both (steps already running finish; the tools pick them up when started again)
#   scripts/start.sh --no-open  don't open the browser
set -euo pipefail
cd "$(dirname "$0")/.."
tools_url="http://127.0.0.1:8124"
app_url="http://127.0.0.1:8125"
mkdir -p runs/service_logs
tools_pid=runs/service_logs/tools.pid
app_pid=runs/service_logs/app.pid   # only when the app runs without a container

# each service starts in its own session (perl's setsid), so closing this terminal doesn't stop it
setsid=(nohup perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV')
have_docker() { command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; }
stop_pid() {
  if [[ -f "$1" ]] && kill -0 "$(cat "$1")" 2>/dev/null; then
    kill "$(cat "$1")"
    for _ in $(seq 1 20); do kill -0 "$(cat "$1")" 2>/dev/null || break; sleep 0.5; done
  fi
  rm -f "$1"
}
wait_for() {  # url, what
  for _ in $(seq 1 60); do curl -sf "$1" >/dev/null 2>&1 && return 0; sleep 1; done
  echo "$2 didn't answer at $1. Log: runs/service_logs/" >&2
  exit 1
}

if [[ "${1:-}" == "--stop" ]]; then
  have_docker && docker compose down
  stop_pid "$app_pid"
  stop_pid "$tools_pid"
  echo "Stopped."
  exit 0
fi

echo "== Tools (native)"
uv sync --extra web --extra report --quiet
scripts/deploy.sh                      # a release of the tools: new work uses it, running work keeps its own
stop_pid "$tools_pid"
"${setsid[@]}" .venv/bin/pruneagent tools-service >> runs/service_logs/tools.log 2>&1 &
echo $! > "$tools_pid"
wait_for "$tools_url/api/service" "The tools"
for _ in $(seq 1 15); do curl -s "$tools_url/api/service" | grep -q '"running":true' && break; sleep 1; done
curl -s "$tools_url/api/service" | grep -q '"running":true' || { echo "The tools' workflow runner didn't start. Log: runs/service_logs/tools.log" >&2; exit 1; }
echo "Tools API: $tools_url"

echo "== App"
if have_docker; then
  stop_pid "$app_pid"
  docker compose up -d --build
else
  echo "Docker isn't running, so the app runs without a container for now."
  [[ -d web/node_modules ]] || (cd web && npm install --silent)
  (cd web && npm run build --silent >/dev/null)
  stop_pid "$app_pid"
  "${setsid[@]}" .venv/bin/pruneagent web >> runs/service_logs/app.log 2>&1 &
  echo $! > "$app_pid"
fi
wait_for "$app_url/api/service" "The app"
echo "App: $app_url"
[[ "${1:-}" == "--no-open" ]] || open "$app_url"
