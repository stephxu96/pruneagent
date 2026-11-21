"""Chart for docs/strategy/finding.md: quiz score per question set, one layer per step vs several layers per step vs
FLAP (Qwen3-0.6B, 20% smaller). Run from the repo root: python docs/strategy/chart.py"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

SETS = {"hellaswag": 10042, "boolq": 3270, "arc_easy": 2376, "winogrande": 1267, "arc_challenge": 1172,
        "openbookqa": 500, "rte": 277}
ONE, SEVERAL = "20261120-1716-qwen3-0.6b-codex", "20261118-1718-qwen3-0.6b-codex"


def tasks(run, who="agent"):
    return json.loads(Path("runs", run, "accuracy.json").read_text())["results"][who]["tasks"]


def weighted(t):
    return 100 * sum(t[s]["acc"] * n for s, n in SETS.items()) / sum(SETS.values())


series = [("One layer per step", tasks(ONE), "#2a6fdb"), ("Several layers per step", tasks(SEVERAL), "#8fb3ee"),
          ("FLAP", tasks(SEVERAL, "Baseline: FLAP"), "#9aa0a6")]
fig, ax = plt.subplots(figsize=(8, 3.8), dpi=150)
w = 0.27
for k, (name, t, color) in enumerate(series):
    ax.bar([i + (k - 1) * w for i in range(len(SETS))], [100 * t[s]["acc"] for s in SETS], w - 0.03,
           label=f"{name} ({weighted(t):.1f}% overall)", color=color)
ax.set_xticks(range(len(SETS)), [f"{s}\n{n:,}" for s, n in SETS.items()], fontsize=8)
ax.set_xlabel("Question set and number of questions")
ax.set_ylabel("Questions right (%)")
ax.set_ylim(0, 75)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25)
ax.legend(frameon=False, loc="upper right", fontsize=8)
ax.set_title("Qwen3-0.6B, 20% smaller: quiz score by strategy (18,904 questions)", fontsize=10)
fig.tight_layout()
fig.savefig(Path(__file__).with_name("strategy.png"))
for name, t, _ in series:
    print(name, round(weighted(t), 2))
