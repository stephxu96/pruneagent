"""llm-compressor Wanda cross-check on a fresh Colab GPU (task 1.1).

    scripts/colab_job.sh pa-t4 T4 scripts/colab_check_wanda.py results/wanda/qwen3-0.6b-base__wanda50__llmcompressor__cuda.json

Clones the latest pushed `main`, installs the locked project (needed to rebuild the data cache), checks the
cache against the committed manifest, then runs scripts/check_wanda_llmcompressor.py in its own uv environment
with llm-compressor 0.8.1. The result JSON is written to /content/result.json for download.
"""

import json
import os
import subprocess
import sys

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

    sh([
        "uv", "run", "--no-project", "--python", "3.12", "--with", "llmcompressor==0.8.1",
        "python", "scripts/check_wanda_llmcompressor.py", "--model", MODEL, "--sparsity", "0.5", "--device", "cuda",
    ])
    src = "results/wanda/qwen3-0.6b-base__wanda50__llmcompressor__cuda.json"
    with open(src) as f:
        result = json.load(f)
    result["git_ref"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    with open("/content/result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
