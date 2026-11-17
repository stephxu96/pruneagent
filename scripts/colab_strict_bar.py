"""FLAP and uniform Wanda-sp under the strict bar (no allowance), on a fresh Colab T4.

    scripts/colab_job.sh pa-t4-strict T4 scripts/colab_strict_bar.py results/baselines_strict/qwen3-0.6b-base__colab.json

Strict bar: validation perplexity <= Wanda-50%'s on the T4 (23.69, task 1.9 run 1) and mean accuracy >= Wanda-50%'s
(0.4380, results/mc/qwen3-0.6b-base__colab.json). Search intervals start from task 1.9's T4 measurements:
- FLAP: perplexity passes up to ~0.25 (23.48 at 0.25, 24.86 at 0.2656), so the perplexity search is a formality;
  accuracy is 0.4304 at 0.1817 (fails), so the accuracy search covers [0.12, 0.1817]. Linear interpolation
  between the unpruned model (0.517) and 0.1817 puts the crossing near 0.165.
- Wanda-sp: perplexity 21.12 at 0.125 and 24.90 at 0.1875 (run 1 log), so the perplexity search covers
  [0.125, 0.1875]; if accuracy fails there, the accuracy search covers [0.12, that ratio].
Accuracy steps stop at 0.02 width (~2 checks of ~12 min each), about the size of accuracy's own margin of error.
Every accuracy check saves which questions were answered correctly, and Wanda-50% is re-run the same way at the
end, so each method can be compared with Wanda-50% question by question (scripts/paired_strict.py).
"""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

REPO = "https://github.com/stephxu96/pruneagent.git"
BRANCH = "strict-bar"
MODEL = "Qwen/Qwen3-0.6B-Base"
RESULTS = "results/baselines_strict"
BAR = ["--max-ppl", "23.69", "--min-acc", "0.438038703166515"]
RUNS = [
    ("flap", ["--lo", "0.25", "--hi", "0.2656", "--tol", "0.02", "--acc-fails-at", "0.1817", "--acc-lo", "0.12",
              "--acc-tol", "0.02"]),
    ("wanda-sp", ["--lo", "0.125", "--hi", "0.1875", "--acc-lo", "0.12", "--acc-tol", "0.02"]),
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
    for folder in (RESULTS,):
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
    stop = threading.Event()

    def autosave() -> None:  # `baseline` rewrites its JSON after every measurement; copy it out every minute
        while not stop.wait(60):
            try:
                save(combined)
            except (OSError, ValueError) as e:  # a half-written file; the next pass picks it up
                print(f"autosave skipped: {e}", file=sys.stderr, flush=True)

    threading.Thread(target=autosave, daemon=True).start()
    try:
        for method, extra in RUNS:
            sh(["uv", "run", "pruneagent", "baseline", "--model", MODEL, "--method", method, "--device", "cuda",
                "--results", RESULTS, "--per-question", *BAR, *extra])
            save(combined)
        sh(["uv", "run", "pruneagent", "mc", "--model", MODEL, "--sparsity", "0.5", "--device", "cuda",
            "--results", RESULTS, "--per-question"])
    finally:
        stop.set()
        save(combined)
    print(json.dumps({k: v.get("search", {}).get("ratio", v.get("mean_acc")) for k, v in combined["runs"].items()},
                     indent=2))


if __name__ == "__main__":
    main()
