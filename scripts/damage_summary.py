"""Chart and chart-ready table for the measured damage table (results/damage_table/summary.md).

    uv run --with matplotlib python scripts/damage_summary.py   (matplotlib is not a project dependency)

Reads results/damage_table/qwen3-0.6b-base__mps__float16.json and writes, next to it, damage_by_layer.png (one
panel per strategy: perplexity rise per layer, log scale) and damage_by_layer.csv; prints the safest and most
damaging layers per strategy and the rank agreement with the quick tests.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullLocator

from pruneagent.plan import STRATEGIES

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "results/damage_table"
SRC = DIR / "qwen3-0.6b-base__mps__float16.json"
FLOOR = 0.01  # % rise: smaller rises (and falls) are drawn as no bar (log scale)
CAP = 1000  # % rise: larger rises are cut off here and labelled with their value
BLUE, DARK, INK, MUTED, SURFACE = "#2a78d6", "#104281", "#0b0b0b", "#52514e", "#fcfcfb"


def pct(v: float) -> str:
    return f"{v:,.0f}%" if abs(v) >= 100 else f"{v:.1f}%" if abs(v) >= 1 else f"{v:.2f}%"


def main() -> None:
    r = json.loads(SRC.read_text())
    e = r["entries"]
    n_layers = 1 + max(v["layer"] for v in e.values())
    layers = list(range(n_layers))
    rise = {s: [e[f"{i}/{s}"]["increase_pct"] for i in layers] for s in STRATEGIES}

    with (DIR / "damage_by_layer.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", *(f"{s}_rise_pct" for s in STRATEGIES), *(f"{s}_MB_freed" for s in STRATEGIES)])
        for i in layers:
            w.writerow([i, *(round(rise[s][i], 3) for s in STRATEGIES),
                        *(round(e[f"{i}/{s}"]["weight_bytes_freed"] / 1e6, 1) for s in STRATEGIES)])

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 4, figsize=(13, 7.2), sharey=True, facecolor=SURFACE)
    for ax, s in zip(axes, STRATEGIES, strict=True):  # one shared scale, so the panels compare directly
        vals = [min(max(v, FLOOR), CAP) for v in rise[s]]
        lo, hi = min(range(n_layers), key=lambda i: rise[s][i]), max(range(n_layers), key=lambda i: rise[s][i])
        ax.barh(layers, vals, height=0.7, color=[DARK if i in (lo, hi) else BLUE for i in layers])
        ax.set_xscale("log")
        ax.set_xlim(FLOOR, CAP * 20)
        ax.set_xticks([0.01, 1, 100])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}%"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_facecolor(SURFACE)
        ax.grid(axis="x", color="#e4e3df", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(s, fontsize=10, color=INK, loc="left", family="monospace")
        for i in layers:  # label the extremes, and every bar that is cut off or below the floor
            if i in (lo, hi) or rise[s][i] > CAP or (rise[s][i] < FLOOR and i == lo):
                label = pct(rise[s][i]) if rise[s][i] >= FLOOR else f"{pct(rise[s][i])} (fell)"
                ax.text(FLOOR * 1.15 if rise[s][i] < FLOOR else vals[i] * 1.2, i, label, va="center",
                        fontsize=8, color=INK)
    axes[0].set_yticks(layers)
    axes[0].set_ylim(n_layers - 0.5, -0.5)
    axes[0].set_ylabel("layer")
    fig.supxlabel("perplexity rise over unpruned (log scale)", fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(DIR / "damage_by_layer.png", dpi=150, facecolor=SURFACE)

    base = r["dense"]["ppl"]
    print(f"unpruned ppl {base:.4f}; {len(e)} entries; Mac GPU {r['gpu_seconds'] / 60:.1f} min")
    for s in STRATEGIES:
        order = sorted(layers, key=lambda i: rise[s][i])
        mb = e[f"0/{s}"]["weight_bytes_freed"] / 1e6
        print(f"{s:24s} frees {mb:5.1f} MB | safest " + ", ".join(f"{i} ({pct(rise[s][i])})" for i in order[:4])
              + " | worst " + ", ".join(f"{i} ({pct(rise[s][i])})" for i in order[::-1][:4]))
    for name, c in r.get("comparison", {}).items():
        print(f"quick tests [{name}]: all pairs {c.get('all', float('nan')):.2f}; "
              + "; ".join(f"{s} {v['spearman']:.2f} (BI {v['spearman_block_influence']:.2f}) gaps "
                          + ",".join(f"L{g['layer']}:{g['probe_rank']:.0f}->{g['ppl_rank']:.0f}"
                                     for g in v["largest_rank_gaps"])
                          for s, v in c["per_strategy"].items()))


if __name__ == "__main__":
    main()
