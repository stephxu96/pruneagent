#!/usr/bin/env bash
# A release of the PruneAgent tools (docs/APP_DESIGN.md §4.3): a fixed copy of the committed code with its own
# environment, in releases/<commit>/. New workflows use it; running ones finish on the release they started with
# (AI instructions included). Called by scripts/start.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -n "$(git status --porcelain -- src scripts pyproject.toml uv.lock)" ]]; then
  echo "Commit first: a release is made from committed code." >&2
  exit 1
fi
sha="$(git rev-parse --short=12 HEAD)"
rel="releases/$sha"
if [[ -f "$rel/.ready" ]]; then
  echo "Tools release $sha already built"
else
  # built in place (an environment records its own path), and marked ready only when complete
  echo "Building tools release $sha"
  rm -rf "$rel" && mkdir -p "$rel"
  git archive HEAD | tar -x -C "$rel"
  (cd "$rel" && uv sync --frozen --extra web --extra report --quiet)
  "$rel/.venv/bin/pruneagent" version >/dev/null
  touch "$rel/.ready"
fi
# point releases/current at it in one step (rename over the old link)
python3 -c "import os, sys; os.symlink(sys.argv[1], 'releases/current.new'); os.replace('releases/current.new', 'releases/current')" "$sha"
echo "New workflows use tools release $sha"
