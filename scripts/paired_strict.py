"""Each method's accuracy at the strict bar minus Wanda-50%'s, on the same questions, with 95% intervals.

    uv run python scripts/paired_strict.py [results/baselines_strict/qwen3-0.6b-base__colab.json]

Reads the Colab job's combined results (per-question correctness saved by `--per-question`), compares every
accuracy check of each method with the Wanda-50% re-run, and writes results/baselines_strict/qwen3-0.6b-base__paired.json.
"""

import json
import sys
from pathlib import Path

from pruneagent import mc

src = Path(sys.argv[1] if len(sys.argv) > 1 else "results/baselines_strict/qwen3-0.6b-base__colab.json")
runs = json.loads(src.read_text())["runs"]
ref_key = next(k for k in runs if "__wanda50__" in k)
ref = runs[ref_key]
out = {"source": str(src), "reference": ref_key, "reference_mean_acc": ref["mean_acc"], "methods": {}}
for run in runs.values():
    if "search" not in run:
        continue
    final = run["search"].get("ratio")
    rows = {}
    for ratio, trial in run["trials"].items():
        if "per_question" not in trial.get("mc", {}):
            continue
        p = mc.paired_difference(trial["mc"]["per_question"], ref["per_question"])
        is_final = final is not None and abs(float(ratio) - final) < 1e-9
        rows[ratio] = {"mean_acc": trial["mc"]["mean_acc"], "weight_bytes_removed": trial["weight_bytes_removed"],
                       "final": is_final, **p}
        m = p["mean"]
        print(f"{run['method']:<9} ratio {ratio}{' (final)' if is_final else '        '}: "
              f"accuracy minus Wanda-50% {100 * m['diff']:+.2f} points "
              f"(95% interval {100 * m['ci95'][0]:+.2f} to {100 * m['ci95'][1]:+.2f})")
        if is_final:
            for task, t in p["tasks"].items():
                print(f"    {task:<14} {100 * t['diff']:+.2f} ({100 * t['ci95'][0]:+.2f} to {100 * t['ci95'][1]:+.2f}),"
                      f" n={t['n']}")
    out["methods"][run["method"]] = {"final_ratio": final, "checks": rows}
dest = src.with_name(src.name.replace("__colab.json", "__paired.json"))
dest.write_text(json.dumps(out, indent=2) + "\n")
print(f"Saved {dest}")
