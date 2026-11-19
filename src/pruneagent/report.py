"""Task 3.5: one run as a plain-language report (Markdown + charts + JSON), from the files the run wrote.

`pruneagent report <run-id>` writes `runs/<run-id>/report/`:
- `report.md`: headline, settings, results against Baseline: FLAP, memory, what the AI did, every round, cost;
- `path.png`, `layers.png`, `rounds.png`: the charts;
- `report.json`: every number in the report, for the final comparison (4.4) and the slides.

It reads the same views as the web app (`web/views.py`), so the report and the app always agree. Wording follows
the app: plain words first, the technical name in brackets once.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from pruneagent.web import views as V

AGENT, FLAP, PART, GOOD, IDEA = "#eb6834", "#2a78d6", "#55636f", "#1a8a5c", "#b7b6b0"
STRATEGY = {"narrow_mlp": "slim a layer", "remove_attention_group": "remove an attention unit",
            "drop_layer": "remove a whole layer"}
OUTCOME = {"ACCEPT": "kept", "ROLLBACK": "nothing kept (every idea hurt quality too much)",
           "INVALID": "ideas refused (repeated something already tried)"}


def _loss(error: float, base: float) -> float:
    return error / base - 1


def _loss_text(x: float) -> str:
    return "no loss" if x < 0.005 else (f"{x + 1:.0f}× worse" if x >= 2 else f"{x:.0%} worse")


def _changes(changes: list[dict]) -> str:
    by: dict[str, list[int]] = {}
    for c in changes:
        by.setdefault(c["strategy"], []).append(c["layer"] + 1)
    parts = []
    for s, layers in by.items():
        ls = sorted(set(layers))
        twice = " (twice)" if len(layers) > len(ls) else ""
        parts.append(f"{STRATEGY.get(s, s)}: layer{'s' if len(ls) > 1 else ''} {', '.join(map(str, ls))}{twice}")
    return "; ".join(parts)


def summarize(root: Path, run_id: str) -> dict:
    """Every number the report shows."""
    d = V.run_detail(root, run_id)
    res = V.run_results(root, run_id)
    base = d["original_error"]
    proposed, kept = Counter(), Counter()
    for r in d["rounds"]:
        for c in r["candidates"]:
            for ch in c["changes"]:
                proposed[ch["strategy"]] += 1
                if r["chosen"] == c["id"]:
                    kept[ch["strategy"]] += 1
    first, last = d["steps"][0], d["steps"][-1]
    out = {
        "run": run_id, "model": d["model"], "state": d["state"], "reason": d["reason"], "settings": d["settings"],
        "original_mb": d["original_mb"], "limit_mb": d["limit_mb"], "cut": d["cut"],
        "final": {"size_mb": last["size_mb"], "memory_requirement_mb": last["mac_mb"],
                  "quality_loss_search": _loss(last["error"], base)},
        "original_memory_requirement_mb": first["mac_mb"],
        "flap": None if not d["flap"] else {
            "size_mb": d["flap"]["size_mb"], "memory_requirement_mb": d["flap"]["mac_mb"],
            "quality_loss_search": _loss(d["flap"]["error"], base) if d["flap"]["error"] else None},
        "rounds": len(d["rounds"]),
        "outcomes": dict(Counter(r["outcome"] for r in d["rounds"])),
        "strategies": {s: {"proposed": proposed[s], "kept": kept[s]} for s in STRATEGY},
        "layers_removed": [i + 1 for i, layer in enumerate(last["layers"]) if layer["dropped"]],
        "cost": {"ai_calls": sum(1 for r in d["rounds"] if r["agent_seconds"] > 0),
                 "ai_seconds": sum(r["agent_seconds"] for r in d["rounds"]),
                 "check_seconds": sum(r["check_seconds"] for r in d["rounds"])},
        "heldout": None, "accuracy": None,
    }
    if res["heldout"]:
        h = res["heldout"]
        o = h["unpruned"]["test_error"]
        out["heldout"] = {name: {"test_error": v["test_error"], "quality_loss": _loss(v["test_error"], o),
                                 "size_mb": v["size_mb"]} for name, v in h.items()}
    if res["accuracy"]:
        out["accuracy"] = res["accuracy"]
    return out


# ---------------------------------------------------------------------------------------------------- charts


def _style(ax, title: str, x: str, y: str) -> None:
    ax.set_title(title, loc="left", fontsize=12)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _charts(d: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    orig, base, goal = d["original_mb"], d["original_error"], d["original_mb"] - d["limit_mb"]

    # 1. every idea tested: memory saved against quality lost
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ideas = [(orig - c["size_mb"], _loss(c["error"], base) * 100) for r in d["rounds"] for c in r["candidates"]
             if c["error"] is not None and c["size_mb"] is not None and r["chosen"] != c["id"]]
    if ideas:
        ax.scatter(*zip(*ideas, strict=True), s=14, color=IDEA, label="ideas tested, not kept", zorder=2)
    path = [(orig - s["size_mb"], _loss(s["error"], base) * 100) for s in d["steps"]]
    ax.plot(*zip(*path, strict=True), color=AGENT, marker="o", markersize=4, linewidth=2,
            label="PruneAgent: ideas kept", zorder=3)
    if d["flap"] and d["flap"]["error"]:
        fy = _loss(d["flap"]["error"], base) * 100
        top = max(p[1] for p in path) * 1.15
        if fy <= top * 1.5:
            ax.scatter([orig - d["flap"]["size_mb"]], [fy], marker="D", s=50, color=FLAP, zorder=4,
                       label="Standard method (FLAP), same size")
        else:
            ax.annotate(f"Standard method (FLAP), same size: {_loss_text(fy / 100)} (off the chart)",
                        xy=(0.98, 0.04), xycoords="axes fraction", color=FLAP, fontsize=9, ha="right")
            ax.set_ylim(top=top)
    ax.axvline(goal, color=GOOD, linestyle="--", linewidth=1.5, label="goal: fits")
    _style(ax, "Every idea tested", "Memory saved (MB)", "Quality lost (%)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "path.png")
    plt.close(fig)

    # 2. what is left of each layer: PruneAgent against FLAP
    n = d["shape"]["layers"]
    full_mlp, full_g = d["shape"]["mlp_full"], d["shape"]["groups_full"]
    last = d["steps"][-1]["layers"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), dpi=150, sharex=True)
    xs = range(1, n + 1)
    w = 0.4
    for ax, key, full, title in ((axes[0], "mlp", full_mlp, "Knowledge part left in each layer (%)"),
                                 (axes[1], "groups", full_g, "Attention units left in each layer")):
        scale = 100 / full if key == "mlp" else 1
        agent = [0 if layer["dropped"] else layer[key] * scale for layer in last]
        ax.bar([x - w / 2 for x in xs], agent, width=w, color=AGENT, label="PruneAgent")
        if d["flap"]:
            fl = [0 if layer["dropped"] else layer[key] * scale for layer in d["flap"]["layers"]]
            ax.bar([x + w / 2 for x in xs], fl, width=w, color=FLAP, label="Standard method (FLAP)")
        for i, layer in enumerate(last):
            if layer["dropped"]:
                ax.text(i + 1 - w / 2, 0, "✕", ha="center", va="bottom", color=AGENT, fontsize=9)
        _style(ax, title, "", "")
        ax.set_ylim(0, (100 if key == "mlp" else full_g) * 1.1)
    axes[1].set_xlabel("Layer (✕ = removed entirely)")
    axes[1].set_xticks(list(xs))
    axes[0].legend(frameon=False, fontsize=9, ncols=2, loc="lower right", bbox_to_anchor=(1, 1.0))
    fig.tight_layout()
    fig.savefig(out / "layers.png")
    plt.close(fig)

    # 3. quality lost after each round
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    kept_at = {s["round"]: _loss(s["error"], base) * 100 for s in d["steps"]}
    xs2, ys, cur = [0], [0.0], 0.0
    for r in d["rounds"]:
        cur = kept_at.get(r["round"], cur)
        xs2.append(r["round"])
        ys.append(cur)
    ax.step(xs2, ys, where="post", color=AGENT, linewidth=2, label="PruneAgent: quality lost so far")
    for marker, outcome, label in (("x", "ROLLBACK", "nothing kept"), ("s", "INVALID", "ideas refused")):
        pts = [(r["round"], ys[xs2.index(r["round"])]) for r in d["rounds"] if r["outcome"] == outcome]
        if pts:
            ax.scatter(*zip(*pts, strict=True), marker=marker, color=PART, s=40, zorder=3, label=label)
    _style(ax, "Quality lost after each round", "Round", "Quality lost (%)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "rounds.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------- markdown


def _markdown(s: dict, d: dict) -> str:
    name = s["model"].removeprefix("Qwen/").removesuffix("-Base")
    st = s["settings"]
    L = [f"# {name}, {s['cut']:.0%} smaller: run {s['run']}", ""]
    h = s["heldout"]
    if h and "agent" in h and "Baseline: FLAP" in h:
        a, f = h["agent"], h["Baseline: FLAP"]
        L += [(f"**Same size, on text neither had seen: PruneAgent is {_loss_text(a['quality_loss'])} than the "
               f"original, the standard method (FLAP) {_loss_text(f['quality_loss'])}.**"), ""]
    else:
        fl = s["flap"]["quality_loss_search"] if s["flap"] else None
        L += [f"**Quality lost during the search: PruneAgent {_loss_text(s['final']['quality_loss_search'])}"
              + (f", the standard method (FLAP) {_loss_text(fl)}" if fl is not None else "")
              + ".** The final check on new text has not been run yet (`scripts/heldout_check.py`).", ""]
    L += [f"The model went from {s['original_mb']:,.0f} MB to {s['final']['size_mb']:,.0f} MB "
          f"(goal: under {s['limit_mb']:,.0f} MB) in {s['rounds']} rounds. Status: {s['state']}"
          + (f" ({s['reason']})" if s["reason"] else "") + ".", ""]

    L += ["## Settings", "", "| Setting | Value |", "|---|---|"]
    rows = [("Model", name), ("Original size", f"{s['original_mb']:,.0f} MB"),
            ("Target size", f"under {s['limit_mb']:,.0f} MB ({s['cut']:.0%} smaller)"),
            ("Device", st["device"].upper()), ("AI", st["agent"]), ("Ideas per round", st["ideas_per_round"]),
            ("Max quality drop per step", f"{st['max_drop_per_step']:.0%}"), ("Max rounds", st["max_rounds"]),
            ("Max AI calls", st["max_ai_calls"]), ("Max layers removed", st["max_layers_removed"]),
            ("Min layer width kept", f"{st['min_layer_width']:.0%}"), ("Sample text", st["sample_text"])]
    L += [f"| {k} | {v} |" for k, v in rows] + [""]

    L += ["## Results", ""]
    if h:
        L += ["Quality on text neither model saw during the search (lower prediction error is better).", "",
              "| Model | Size | Prediction error (perplexity) | Quality lost |", "|---|---:|---:|---:|"]
        label = {"unpruned": "Original", "agent": "**PruneAgent**", "Baseline: FLAP": "Standard method (FLAP)"}
        for k in ("unpruned", "agent", "Baseline: FLAP"):
            if k in h:
                v = h[k]
                L.append(f"| {label[k]} | {v['size_mb']:,.0f} MB | {v['test_error']:.2f} | "
                         f"{'—' if k == 'unpruned' else _loss_text(v['quality_loss'])} |")
        L.append("")
    acc = s["accuracy"]
    if acc and "agent" in acc and "Baseline: FLAP" in acc:
        a, f = acc["agent"]["all_questions"], acc["Baseline: FLAP"]["all_questions"]
        gap, se = a["acc"] - f["acc"], (a["stderr"] ** 2 + f["stderr"] ** 2) ** 0.5
        clear = "a clear difference" if abs(gap) >= 2 * se else "too close to call"
        L += [(f"Quiz questions ({a['questions']:,}, each counted once): PruneAgent {a['acc']:.1%} right, "
               f"the standard method {f['acc']:.1%} ({gap * 100:+.1f} points, {clear})."), "",
              "| Question set | PruneAgent | Standard method (FLAP) | Difference | Clear? |", "|---|---:|---:|---:|---|"]
        for t, v in acc["agent"]["sets"].items():
            w = acc["Baseline: FLAP"]["sets"][t]
            g = v["acc"] - w["acc"]
            c = abs(g) >= 2 * ((v["acc_stderr"] or 0) ** 2 + (w["acc_stderr"] or 0) ** 2) ** 0.5
            L.append(f"| {t} | {v['acc']:.1%} | {w['acc']:.1%} | {g * 100:+.1f} | {'yes' if c else 'no'} |")
        L.append("")
    L += ["![Every idea tested](path.png)", "",
          ("*Each grey dot is an idea the AI suggested and the computer tested; the orange path is the ideas "
           "kept. The dashed line is the goal.*"), ""]

    L += ["## Memory", "", "| | Size | Memory requirement (MAC) |", "|---|---:|---:|",
          f"| Original | {s['original_mb']:,.0f} MB | {s['original_memory_requirement_mb']:,.0f} MB |",
          f"| **PruneAgent** | {s['final']['size_mb']:,.0f} MB | {s['final']['memory_requirement_mb']:,.0f} MB |"]
    if s["flap"]:
        L.append(f"| Standard method (FLAP) | {s['flap']['size_mb']:,.0f} MB | "
                 f"{s['flap']['memory_requirement_mb']:,.0f} MB |")
    L += ["", (f"*Memory requirement: the model plus its notes on a {d['mac_context']}-token conversation; "
               "all of it is read once for every token written (memory access cost, MAC).*"), ""]

    L += ["## What the AI did", "",
          "| Kind of cut | Suggested | Kept |", "|---|---:|---:|"]
    L += [f"| {STRATEGY[k]} | {v['proposed']} | {v['kept']} |" for k, v in s["strategies"].items()]
    removed = s["layers_removed"]
    L += ["", f"Layers removed entirely: {', '.join(map(str, removed)) if removed else 'none'}.", "",
          "![What is left of each layer](layers.png)", "",
          "![Quality lost after each round](rounds.png)", ""]

    L += ["## Every round", "", "| Round | Outcome | Kept idea | Quality lost after |", "|---|---|---|---:|"]
    kept_at = {st_["round"]: st_ for st_ in d["steps"]}
    cur = d["steps"][0]
    for r in d["rounds"]:
        cur = kept_at.get(r["round"], cur)
        chosen = next((c for c in r["candidates"] if c["id"] == r["chosen"]), None)
        L.append(f"| {r['round']} | {OUTCOME.get(r['outcome'], r['outcome'] or '—')} | "
                 f"{_changes(chosen['changes']) if chosen else '—'} | "
                 f"{_loss_text(_loss(cur['error'], d['original_error']))} |")
    L += ["", "<details><summary>The AI's reasons, round by round</summary>", ""]
    for r in d["rounds"]:
        L.append(f"- **Round {r['round']}:** " + " ".join(
            f"Idea {c['id'] + 1} ({_changes(c['changes'])}): {c['why']}" for c in r["candidates"])
                 if r["candidates"] else f"- **Round {r['round']}:** {' '.join(r['refused_reasons'])}")
    L += ["", "</details>", ""]

    c = s["cost"]
    L += ["## Cost", "", (f"{c['ai_calls']} AI calls ({c['ai_seconds'] / 60:.0f} minutes waiting for the AI), "
                          f"{c['check_seconds'] / 60:.0f} minutes testing ideas on the {st['device'].upper()}."), ""]
    return "\n".join(L)


def write(root: Path, run_id: str, out_dir: Path | None = None) -> Path:
    out = Path(out_dir) if out_dir else Path(root) / run_id / "report"
    out.mkdir(parents=True, exist_ok=True)
    d = V.run_detail(Path(root), run_id)
    s = summarize(Path(root), run_id)
    _charts(d, out)
    (out / "report.md").write_text(_markdown(s, d))
    (out / "report.json").write_text(json.dumps(s, indent=1))
    return out / "report.md"
