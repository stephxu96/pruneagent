"""Layer probes vs. full-model perplexity on a fresh Colab T4 (task 1.8, assumption A1).

    scripts/colab_job.sh pa-t4-probe T4 scripts/colab_probe.py results/probes/qwen3-0.6b-base__cuda__float16.json

Clones branch `task-1.8`, installs the locked project, rebuilds the data cache and checks it against the
committed manifest, computes the pruning scores on the VM, then runs `pruneagent probe --ground-truth`: the probe
table (1-3 steps) and full validation perplexity (128 windows) after one step of each strategy on each of the 28
layers. The results JSON is rewritten after every evaluation at /content/result.json, so a cut-off run still
leaves the evaluations done so far.
"""

import json
import os
import subprocess
import sys

REPO = "https://github.com/stephxu96/pruneagent.git"
BRANCH = "task-1.8"
MODEL = "Qwen/Qwen3-0.6B-Base"
OUT = "/content/result.json"


def sh(cmd: list[str]) -> None:
    """Run a command, forwarding output line by line (sys.stderr has no file descriptor in a Colab kernel)."""
    print("$", " ".join(cmd), file=sys.stderr, flush=True)
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        for line in proc.stdout:
            print(line, end="", file=sys.stderr, flush=True)
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def main() -> None:
    os.chdir("/content")
    sh(["git", "clone", "-q", "--branch", BRANCH, REPO, "pruneagent"])
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

    sh(["uv", "run", "pruneagent", "probe", "--model", MODEL, "--device", "cuda", "--steps", "3",
        "--ground-truth", "--out", OUT])
    git_ref = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    with open(OUT) as f:
        result = json.load(f)
    result["git_ref"] = git_ref
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["comparison"], indent=2))


if __name__ == "__main__":
    main()
