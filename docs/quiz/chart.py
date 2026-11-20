"""Quiz chart for docs/quiz/finding.md: share of questions right, agent vs FLAP, from each run's accuracy.json.
Run from the repo root: .venv/bin/python docs/quiz/chart.py"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

SETS = {"boolq": 3270, "rte": 277, "hellaswag": 10042, "winogrande": 1267, "arc_easy": 2376,
        "arc_challenge": 1172, "openbookqa": 500}
RUNS = [("20261118-0046-qwen3-0.6b-codex", "0.6B\n10% smaller"), ("20261118-1718-qwen3-0.6b-codex", "0.6B\n20% smaller"),
        ("20261118-1016-qwen3-0.6b-codex", "0.6B\n30% smaller"), ("20261119-2133-qwen3-1.7b-codex", "1.7B\n10% smaller")]


def weighted(tasks: dict) -> float:
    return 100 * sum(tasks[s]["acc"] * n for s, n in SETS.items()) / sum(SETS.values())


rows = []
for run, label in RUNS:
    r = json.loads(Path("runs", run, "accuracy.json").read_text())["results"]
    rows.append((label, weighted(r["agent"]["tasks"]), weighted(r["Baseline: FLAP"]["tasks"])))

fig, ax = plt.subplots(figsize=(7, 3.8), dpi=150)
x = range(len(rows))
w = 0.36
for off, i, name, color in [(-w / 2, 1, "Agent", "#2a6fdb"), (w / 2, 2, "FLAP", "#9aa0a6")]:
    bars = ax.bar([p + off for p in x], [r[i] for r in rows], w - 0.03, label=name, color=color)
    ax.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=2)
ax.set_xticks(list(x), [r[0] for r in rows])
ax.set_ylabel("Questions right (%)")
ax.set_xlabel("Model and how much smaller")
ax.set_ylim(0, 65)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25)
ax.legend(frameon=False, loc="upper left")
ax.set_title("Quiz score at the same memory limit (18,904 questions)", fontsize=10)
fig.tight_layout()
fig.savefig(Path(__file__).with_name("quiz.png"))
for r in rows:
    print(r)
