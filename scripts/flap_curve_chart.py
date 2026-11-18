"""Chart: test perplexity against memory cut, Baseline: FLAP (curve) and the agent's runs (points).

    uv run python scripts/flap_curve_chart.py
Reads results/flap_curve/qwen3-0.6b-base.json and the agent runs' heldout.json; writes results/flap_curve/*.png/csv.
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

OUT = Path("results/flap_curve")
AGENT_RUNS = {0.10: "runs/20261118-0046-qwen3-0.6b-codex", 0.30: "runs/20261118-1016-qwen3-0.6b-codex"}
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"


def main() -> None:
    flap = json.loads((OUT / "qwen3-0.6b-base.json").read_text())["results"]
    fx = [round(v["memory_cut"] * 100, 1) for v in flap.values()]
    fy = [v["test_ppl"] for v in flap.values()]
    agent = {cut: json.loads((Path(r) / "heldout.json").read_text())["results"] for cut, r in AGENT_RUNS.items()}
    ax_, ay = [c * 100 for c in agent], [a["agent"]["test_ppl"] for a in agent.values()]
    unpruned = agent[0.10]["unpruned"]["test_ppl"]

    with open(OUT / "test_ppl_by_cut.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "memory_cut_pct", "test_ppl"])
        w.writerow(["Unpruned", 0, round(unpruned, 2)])
        w.writerows(["Baseline: FLAP", x, round(y, 2)] for x, y in zip(fx, fy, strict=True))
        w.writerows(["Agent", x, round(y, 2)] for x, y in zip(ax_, ay, strict=True))

    plt.rcParams.update({"font.family": "Helvetica", "font.size": 12})
    fig, ax = plt.subplots(figsize=(9, 5.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhline(unpruned, color=MUTED, lw=1.5, ls=(0, (4, 3)), zorder=1)
    ax.text(33.8, unpruned * 1.04, "Unpruned", color=INK2, va="bottom", ha="right")
    ax.plot(fx, fy, color=BLUE, lw=2, zorder=2)
    ax.scatter(fx, fy, s=64, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3, label="Baseline: FLAP")
    ax.scatter(ax_, ay, s=110, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=4, label="Agent")
    for x, y in zip(ax_, ay, strict=True):
        ax.annotate(f"{y:.1f}", (x, y), xytext=(10, -6), textcoords="offset points", color=INK, fontsize=12)
    for x, y in zip(fx, fy, strict=True):
        if round(x) in (10, 30):
            ax.annotate(f"{y:.1f}", (x, y), xytext=(-6, 10), textcoords="offset points", color=INK, fontsize=12,
                        ha="right" if round(x) == 10 else "left")
    ax.set_yscale("log")
    ticks = [10, 20, 50, 100, 200, 500]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels([str(t) for t in ticks])
    ax.set_ylim(9, 650)
    ax.set_xlim(8, 34)
    ax.set_xticks([10, 15, 20, 25, 30])
    ax.set_xticklabels(["10%", "15%", "20%", "25%", "30%"])
    ax.set_xlabel("Memory cut (share of weight memory removed)", color=INK2)
    ax.set_ylabel("Test perplexity (lower is better)", color=INK2)
    ax.set_title("Quality after pruning Qwen3-0.6B-Base, by share of memory removed", color=INK, loc="left", fontsize=14, pad=12)
    ax.grid(axis="y", color=GRID, lw=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "test_ppl_by_cut.png", dpi=200, facecolor=SURFACE)


if __name__ == "__main__":
    main()
