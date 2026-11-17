"""Gate subset vs. full validation split on a fresh Colab T4 (task 1.12, assumption A9).

    scripts/colab_job.sh pa-t4-subset T4 scripts/colab_subset.py results/subset/qwen3-0.6b-base__cuda__float16.json

Clones the pushed `task-1.12` branch, installs the locked project, rebuilds the data cache and checks it against
the committed manifest, computes pruning scores on the VM, then runs scripts/subset_check.py on every plan in
plans/subset_check/. The result goes to /content/result.json.
"""

import json
import os
import subprocess
import sys

REPO = "https://github.com/stephxu96/pruneagent.git"
BRANCH = "task-1.12"
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

    sh(["uv", "run", "python", "scripts/subset_check.py", "--model", MODEL, "--device", "cuda", "--out", OUT])
    git_ref = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    with open(OUT) as f:
        result = json.load(f)
    result["git_ref"] = git_ref
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["comparison"], indent=2))


if __name__ == "__main__":
    main()
