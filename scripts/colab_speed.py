"""Speed meter on a fresh Colab T4 (task 1.6): device test, unpruned model and mixed plan, two invocations each.

    scripts/colab_job.sh pa-t4-speed T4 scripts/colab_speed.py results/speed/colab_t4.json

Clones the `task-1.6` branch (BRANCH below), installs the locked project, rebuilds the data cache and checks it
against the committed manifest, then runs scripts/speed_session.sh t4 cuda (pruning scores are computed on the
VM). Every JSON written under results/speed goes into /content/result.json.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/stephxu96/pruneagent.git"
BRANCH = "task-1.6"
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
    sh(["git", "clone", "-q", "--branch", BRANCH, REPO, "pruneagent"])
    os.chdir("/content/pruneagent")
    sh([sys.executable, "-m", "pip", "install", "-q", "uv"])
    sh(["uv", "sync", "--locked"])
    sh(["nvidia-smi", "--query-gpu=name,memory.total,clocks.max.sm,clocks.max.mem", "--format=csv,noheader"])

    with open("data_cache/manifest.json") as f:
        committed = json.load(f)["entries"]
    sh(["uv", "run", "pruneagent", "data", "--model", MODEL])
    with open("data_cache/manifest.json") as f:
        rebuilt = json.load(f)["entries"]
    if any(rebuilt[k]["sha256"] != committed.get(k, {}).get("sha256") for k in rebuilt if k.endswith("-base")):
        sys.exit("Data cache on the VM differs from the committed manifest")

    git_ref = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    sh(["bash", "scripts/speed_session.sh", "t4", "cuda"])
    combined = {"git_ref": git_ref, "files": {}}
    for path in sorted(Path("results/speed").rglob("*.json")):
        if "__cuda__" in path.name or path.name == "t4.json":
            combined["files"][str(path)] = json.loads(path.read_text())
    for path in sorted(Path("results/pruned").glob("*mixed_check__cuda*.json")):
        combined["files"][str(path)] = json.loads(path.read_text())
    with open("/content/result.json", "w") as f:
        json.dump(combined, f, indent=2)
    print(json.dumps({k: {m: v.get(m, {}).get("median") if isinstance(v.get(m), dict) else v.get(m)
                          for m in ("prefill_tok_s", "decode_tok_s", "bandwidth_gbs", "compute_tflops")}
                      for k, v in combined["files"].items()}, indent=2))


if __name__ == "__main__":
    main()
