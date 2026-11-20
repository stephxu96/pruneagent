#!/usr/bin/env bash
# Update PruneAgent without touching running work (docs/APP_DESIGN.md §4.3). Run it in the main folder after merging:
#   scripts/deploy.sh
# 1. A new release: a fixed copy of the committed code with its own environment, in releases/<commit>/. New
#    workflows use it; running ones finish on the release they started with (AI instructions included).
# 2. The interface: built into a new folder, then swapped in.
# 3. The web app and the workflow service restart, if they run as background services (scripts/install_services.sh).
#    Steps run as their own processes and write their own exit codes, so the service picks them up again.
set -euo pipefail
cd "$(dirname "$0")/.."
root="$PWD"

if [[ -n "$(git status --porcelain -- src scripts pyproject.toml uv.lock)" ]]; then
  echo "Commit first: a release is made from committed code." >&2
  exit 1
fi
sha="$(git rev-parse --short=12 HEAD)"
rel="releases/$sha"
if [[ -d "$rel" ]]; then
  echo "Release $sha already built"
else
  echo "Building release $sha"
  rm -rf "$rel.partial" && mkdir -p "$rel.partial"
  git archive HEAD | tar -x -C "$rel.partial"
  (cd "$rel.partial" && uv sync --frozen --extra web --extra report --quiet)
  mv "$rel.partial" "$rel"
fi
# point releases/current at it in one step (rename over the old link)
python3 -c "import os, sys; os.symlink(sys.argv[1], 'releases/current.new'); os.replace('releases/current.new', 'releases/current')" "$sha"
echo "New workflows use release $sha"

echo "Building the interface"
(cd web && npx vite build --outDir dist.next --emptyOutDir --logLevel warn)
rm -rf web/dist.old
[[ -d web/dist ]] && mv web/dist web/dist.old
mv web/dist.next web/dist
rm -rf web/dist.old

uid="$(id -u)"
for svc in worker web; do
  if launchctl print "gui/$uid/com.pruneagent.$svc" >/dev/null 2>&1; then
    launchctl kickstart -k "gui/$uid/com.pruneagent.$svc"
    echo "Restarted the $svc service"
  else
    echo "The $svc service isn't installed as a background service; restart it yourself if it runs (scripts/install_services.sh installs both)"
  fi
done
