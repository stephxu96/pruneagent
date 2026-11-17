"""Multiple-choice accuracy, unpruned and Wanda-50%, on a fresh Colab GPU (task 1.1).

    scripts/colab_job.sh pa-t4 T4 scripts/colab_mc.py results/mc/qwen3-0.6b-base__colab.json

Clones the latest pushed `main`, installs the locked project, rebuilds the data cache and checks it against the
committed manifest, then runs `pruneagent mc` at sparsity 0 and 0.5. Both results go into /content/result.json.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/stephxu96/pruneagent.git"
MODEL = "Qwen/Qwen3-0.6B-Base"


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
    sh(["git", "clone", "-q", REPO, "pruneagent"])
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
    for sparsity in ("0", "0.5"):
        sh(["uv", "run", "pruneagent", "mc", "--model", MODEL, "--sparsity", sparsity, "--device", "cuda"])
    for path in sorted(Path("results/mc").glob("*__cuda__*.json")):
        combined["runs"][path.name] = json.loads(path.read_text())
    with open("/content/result.json", "w") as f:
        json.dump(combined, f, indent=2)
    print(json.dumps({k: v["mean_acc"] for k, v in combined["runs"].items()}, indent=2))


if __name__ == "__main__":
    main()
