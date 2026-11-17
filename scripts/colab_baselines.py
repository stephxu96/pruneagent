"""FLAP and uniform Wanda-sp pruned to the quality bar, on a fresh Colab T4 (task 1.9).

    scripts/colab_job.sh pa-t4-bar T4 scripts/colab_baselines.py results/baselines/qwen3-0.6b-base__colab.json

Clones branch `task-1.9`, installs the locked project, rebuilds the data cache and checks it against the committed
manifest, then runs `pruneagent baseline` for both methods. Everything goes into /content/result.json.

Run 2. Run 1 (results/baselines/qwen3-0.6b-base__colab_run1.json, commit d5be693) measured Wanda-50% on the T4
(validation 23.69, for reference; the bar is task 1.1's numbers) and bisected both methods on perplexity over
[0, 0.5]; it found FLAP failing accuracy at 0.2334 and lost the Wanda-sp record to a bookkeeping bug after its
accuracy check. To fit one session, this run starts each search from run 1's final perplexity interval
(FLAP 0.2734-0.2773, Wanda-sp 0.2031-0.2070) and bisects FLAP on the whole bar over [0.13, 0.2334].
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/stephxu96/pruneagent.git"
BRANCH = "task-1.9"
MODEL = "Qwen/Qwen3-0.6B-Base"
RUNS = [
    ("wanda-sp", ["--lo", "0.2031", "--hi", "0.2070"]),
    ("flap", ["--lo", "0.2734", "--hi", "0.2773", "--acc-fails-at", "0.2334", "--acc-lo", "0.13",
              "--acc-tol", "0.0065"]),
]


def sh(cmd: list[str]) -> None:
    """Run a command, forwarding output line by line (sys.stderr has no file descriptor in a Colab kernel)."""
    print("$", " ".join(cmd), file=sys.stderr, flush=True)
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        for line in proc.stdout:
            print(line, end="", file=sys.stderr, flush=True)
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def save(combined: dict) -> None:
    for folder in ("results/baselines", "results/wanda"):
        for path in sorted(Path(folder).glob("*__cuda__*.json")):
            combined["runs"][f"{folder}/{path.name}"] = json.loads(path.read_text())
    with open("/content/result.json", "w") as f:
        json.dump(combined, f, indent=2)


def main() -> None:
    os.chdir("/content")
    sh(["git", "clone", "-q", "-b", BRANCH, REPO, "pruneagent"])
    os.chdir("/content/pruneagent")
    sh([sys.executable, "-m", "pip", "install", "-q", "uv"])
    sh(["uv", "sync", "--locked"])
    sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])

    with open("data_cache/manifest.json") as f:
        committed = json.load(f)["entries"]
    sh(["uv", "run", "pruneagent", "data", "--model", MODEL])
    with open("data_cache/manifest.json") as f:
        rebuilt = json.load(f)["entries"]
    if any(rebuilt[k]["sha256"] != committed.get(k, {}).get("sha256") for k in rebuilt if k.endswith("-base")):
        sys.exit("Data cache on the VM differs from the committed manifest")

    git_ref = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    combined = {"git_ref": git_ref, "runs": {}}
    try:
        for method, extra in RUNS:
            sh(["uv", "run", "pruneagent", "baseline", "--model", MODEL, "--method", method, "--device", "cuda",
                *extra])
            save(combined)
    finally:
        save(combined)
    print(json.dumps({k: v.get("search", {}).get("ratio") for k, v in combined["runs"].items()}, indent=2))


if __name__ == "__main__":
    main()
