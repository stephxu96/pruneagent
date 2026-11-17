"""Dense WikiText-2 perplexity on a fresh Colab GPU (task 0.4, T4 half).

Run from the repo root on the Mac:

    scripts/colab_job.sh pa-t4 T4 scripts/colab_dense_ppl.py results/dense/qwen3-0.6b-base__cuda__float16.json

(runs the latest pushed `main`). The final JSON is also written to /content/result.json on the VM for download.

The VM clones the public repo at <git-ref>, installs the locked dependencies with uv, rebuilds the data
cache and checks it against the committed manifest (same tokens as on the Mac), then runs `pruneagent ppl`.
Only the final results JSON goes to stdout; progress goes to stderr. `colab run` releases the VM afterwards.
"""

import json
import os
import subprocess
import sys

REPO = "https://github.com/stephxu96/pruneagent.git"
MODEL = "Qwen/Qwen3-0.6B-Base"


def sh(cmd: list[str]) -> None:
    """Run a command, forwarding its output line by line to stderr.

    `colab run` executes this file in a Jupyter kernel, where sys.stderr has no file descriptor,
    so output is piped through Python instead of handed to the child process directly.
    """
    print("$", " ".join(cmd), file=sys.stderr, flush=True)
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        for line in proc.stdout:
            print(line, end="", file=sys.stderr, flush=True)
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def main() -> None:
    ref = sys.argv[1] if len(sys.argv) > 1 else "main"
    os.chdir("/content")
    sh(["git", "clone", "-q", REPO, "pruneagent"])
    os.chdir("/content/pruneagent")
    sh(["git", "checkout", "-q", ref])
    sh([sys.executable, "-m", "pip", "install", "-q", "uv"])
    sh(["uv", "sync", "--locked"])
    sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])

    with open("data_cache/manifest.json") as f:
        committed = json.load(f)["entries"]
    sh(["uv", "run", "pruneagent", "data", "--model", MODEL])
    with open("data_cache/manifest.json") as f:
        rebuilt = json.load(f)["entries"]
    data_check = {k: rebuilt[k]["sha256"] == committed.get(k, {}).get("sha256") for k in rebuilt if k.endswith("-base")}

    sh(["uv", "run", "pruneagent", "ppl", "--model", MODEL, "--dtype", "float16"])
    with open("results/dense/qwen3-0.6b-base__cuda__float16.json") as f:
        result = json.load(f)
    result["git_ref"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    result["data_matches_committed_manifest"] = data_check
    with open("/content/result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    if not all(data_check.values()):
        sys.exit("Data cache on the VM differs from the committed manifest")


if __name__ == "__main__":
    main()
