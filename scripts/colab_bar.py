"""Quality bar for one model on a fresh Colab GPU: unpruned and Wanda-50% perplexity and multiple-choice accuracy (task 1.10).

    scripts/colab_job.sh pa-a100 A100 scripts/colab_bar.py results/bar/qwen3-1.7b-base__colab.json

Clones BRANCH, installs the locked project, rebuilds the data cache for MODEL and checks it against the committed
manifest, then runs `pruneagent ppl`, `wanda --sparsity 0.5`, `mc --sparsity 0` and `mc --sparsity 0.5`, all on
cuda. Every results JSON goes into /content/result.json, keyed by its usual path under results/.
`colab exec` runs this file in a kernel, so MODEL and BRANCH are constants rather than arguments.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/stephxu96/pruneagent.git"
BRANCH = "task-1.10"
MODEL = "Qwen/Qwen3-1.7B-Base"
SLUG = MODEL.split("/")[-1].lower()


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
    mine = [k for k in rebuilt if k.endswith("__" + SLUG)]
    if len(mine) < 4 or any(rebuilt[k]["sha256"] != committed.get(k, {}).get("sha256") for k in mine):
        sys.exit(f"Data cache on the VM differs from the committed manifest for {SLUG}")
    print(f"data cache matches the committed manifest: {mine}", file=sys.stderr, flush=True)

    git_ref = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    combined = {"git_ref": git_ref, "model": MODEL, "runs": {}}
    steps = [
        ["ppl"],
        ["wanda", "--sparsity", "0.5"],
        ["mc", "--sparsity", "0"],
        ["mc", "--sparsity", "0.5"],
    ]
    for step in steps:
        sh(["uv", "run", "pruneagent", *step, "--model", MODEL, "--device", "cuda"])
        # Save after every step so a later failure still leaves earlier results to download.
        for folder in ("dense", "wanda", "mc"):
            for path in sorted(Path("results", folder).glob(f"{SLUG}__*cuda__*.json")):
                combined["runs"][f"{folder}/{path.name}"] = json.loads(path.read_text())
        with open("/content/result.json", "w") as f:
            json.dump(combined, f, indent=2)
    print(json.dumps(sorted(combined["runs"]), indent=2))


if __name__ == "__main__":
    main()
