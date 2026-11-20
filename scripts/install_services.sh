#!/usr/bin/env bash
# Run the web app and the workflow service in the background on this Mac, started at login and restarted if they
# stop (macOS launch agents; docs/APP_DESIGN.md §4.1). Remove them with: scripts/install_services.sh --remove
set -euo pipefail
cd "$(dirname "$0")/.."
root="$PWD"
uid="$(id -u)"
agents="$HOME/Library/LaunchAgents"
mkdir -p "$agents" runs/service_logs

for svc in worker web; do
  label="com.pruneagent.$svc"
  plist="$agents/$label.plist"
  launchctl bootout "gui/$uid/$label" 2>/dev/null || true
  if [[ "${1:-}" == "--remove" ]]; then
    rm -f "$plist"
    echo "Removed $label"
    continue
  fi
  if [[ "$svc" == web ]]; then args="<string>web</string><string>--port</string><string>8125</string>"; else args="<string>worker</string>"; fi
  cat > "$plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key>
  <array><string>$root/.venv/bin/pruneagent</string>$args</array>
  <key>WorkingDirectory</key><string>$root</string>
  <key>EnvironmentVariables</key>
  <dict><key>PATH</key><string>$(dirname "$(command -v uv)"):/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string></dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$root/runs/service_logs/$svc.log</string>
  <key>StandardErrorPath</key><string>$root/runs/service_logs/$svc.log</string>
</dict>
</plist>
EOF
  launchctl bootstrap "gui/$uid" "$plist"
  echo "Installed and started $label"
done
