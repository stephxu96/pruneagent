"""Does the gate's 32-window subset rank plans like the full validation split? (task 1.12, assumption A9)

    uv run python scripts/subset_check.py --device cuda --out results/subset/qwen3-0.6b-base__cuda__float16.json

Prunes the unpruned model to every plan in plans/subset_check/, evaluates all 128 validation windows once,
and reads the 32- and 64-window perplexities off the same per-window losses (each window is scored
independently, so this equals evaluating the subset alone; checked directly on the unpruned model and the
first plan with `subset_perplexity`). Reports Spearman and Kendall rank correlation of subset vs. full, and
how many accept/reject decisions at the quality bar would differ.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch

from pruneagent import data as d
from pruneagent import evaluate as ev
from pruneagent import flap, pruner
from pruneagent import plan as P

FIXED_BAR = 26.05  # 1.10 x Wanda-50% validation perplexity (23.68, llm-compressor reference)
BAR_FACTOR = 1.10
REFERENCE = "p00_wanda50_all"


def ranks(x: list[float]) -> list[float]:
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    return cov / math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))


def kendall_tau_b(a: list[float], b: list[float]) -> float:
    conc = disc = ties_a = ties_b = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 and db == 0:
                continue
            if da == 0:
                ties_a += 1
            elif db == 0:
                ties_b += 1
            elif da * db > 0:
                conc += 1
            else:
                disc += 1
    return (conc - disc) / math.sqrt((conc + disc + ties_a) * (conc + disc + ties_b))


def discordant_pairs(names: list[str], a: list[float], b: list[float]) -> list[list[str]]:
    return [[names[i], names[j]] for i in range(len(a)) for j in range(i + 1, len(a))
            if (a[i] - a[j]) * (b[i] - b[j]) < 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=d.DEFAULT_MODEL)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--plans", type=Path, default=Path("plans/subset_check"))
    ap.add_argument("--only", nargs="*", help="Plan names to run (default: all).")
    ap.add_argument("--max-windows", type=int, default=None, help="Debug: evaluate only the first N windows.")
    ap.add_argument("--scores", type=Path, default=None, help="Cached scores (computed if missing).")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = args.device or ev.pick_device()
    manifest = d.read_manifest()
    revision = manifest["entries"][d.calib_name(args.model)]["tokenizer"]["revision"]
    val_name = d.wikitext_name("validation", args.model)
    ids = d.load_tokens(val_name).reshape(-1)
    if args.max_windows:
        ids = ids[: args.max_windows * args.seqlen]
    n_total = ids.numel() // args.seqlen
    sizes = [n for n in (ev.GATE_WINDOWS, 64) if n < n_total] + [n_total]
    subsets = {n: ev.subset_indices(n_total, n) for n in sizes}
    print(f"{args.model}@{revision[:8]} on {device} in {args.dtype}; {n_total} windows; "
          f"subsets {[len(s) for s in subsets.values()]}", flush=True)

    net = ev.load_model(args.model, revision, args.dtype, device)
    start = time.perf_counter()
    if args.scores and args.scores.exists():
        scores = pruner.load_scores(args.scores)
    else:
        meta = {"model": args.model, "revision": revision, "device": device, "dtype": args.dtype,
                "flap_calib": d.flap_calib_name(args.model), "wanda_calib": d.calib_name(args.model)}
        scores = pruner.compute_scores(net, d.load_tokens(d.flap_calib_name(args.model)),
                                       d.load_tokens(d.calib_name(args.model)), device, meta=meta)
        if args.scores:
            args.scores.parent.mkdir(parents=True, exist_ok=True)
            pruner.save_scores(scores, args.scores)
    scores_seconds = round(time.perf_counter() - start, 1)
    print(f"  scores ready in {scores_seconds} s", flush=True)

    def evaluate(model, name: str, direct_check: bool) -> dict:
        r = ev.perplexity(model, ids, args.seqlen, device)
        row = {"ppl": {str(n): ev.ppl_from_losses(r.window_losses, idx) for n, idx in subsets.items()},
               "seconds_full": round(r.seconds, 1), "window_losses": r.window_losses}
        if direct_check:
            sub = ev.subset_perplexity(model, ids, ev.GATE_WINDOWS, args.seqlen, device)
            row["direct_subset_ppl"] = sub.ppl
            row["direct_subset_seconds"] = round(sub.seconds, 1)
        print(f"  {name:28s} " + "  ".join(f"{n}w {v:8.3f}" for n, v in row["ppl"].items())
              + (f"  (direct {ev.GATE_WINDOWS}w {row['direct_subset_ppl']:.3f})" if direct_check else ""),
              flush=True)
        return row

    rows = {"dense": evaluate(net, "dense", direct_check=True)}
    files = sorted(args.plans.glob("*.json"))
    if args.only:
        files = [f for f in files if f.stem in args.only]
    for k, f in enumerate(files):
        spec = json.loads(f.read_text())
        layer_plan = P.apply_steps(P.validate(spec.get("plan", {}), net.config), spec.get("steps", []),
                                   net.config)
        pruned = pruner.prune(net, layer_plan, scores)
        removed = 1 - flap.block_params(pruned) / flap.block_params(net)
        rows[f.stem] = {"description": spec["description"], "block_weights_removed": removed,
                        **evaluate(pruned, f.stem, direct_check=(k == 0))}
        del pruned
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    plans = [n for n in rows if n != "dense"]
    full_key, report = str(n_total), {}
    for n in sizes[:-1]:
        a = [rows[p]["ppl"][str(n)] for p in plans]
        b = [rows[p]["ppl"][full_key] for p in plans]
        flips = [p for p, x, y in zip(plans, a, b, strict=True) if (x <= FIXED_BAR) != (y <= FIXED_BAR)]
        entry = {"spearman": spearman(a, b), "kendall_tau_b": kendall_tau_b(a, b),
                 "discordant_pairs": discordant_pairs(plans, a, b),
                 "fixed_bar": {"bar": FIXED_BAR, "flips": flips}}
        if REFERENCE in rows:
            bar_sub = BAR_FACTOR * rows[REFERENCE]["ppl"][str(n)]
            bar_full = BAR_FACTOR * rows[REFERENCE]["ppl"][full_key]
            rule_flips = [p for p, x, y in zip(plans, a, b, strict=True) if (x <= bar_sub) != (y <= bar_full)]
            entry["rule_bar"] = {"bar_subset": bar_sub, "bar_full": bar_full, "flips": rule_flips}
        report[str(n)] = entry
        print(f"  {n:3d} vs {n_total}: Spearman {entry['spearman']:.4f}  Kendall {entry['kendall_tau_b']:.4f}  "
              f"flips at {FIXED_BAR}: {flips}" + (f"  flips under the §4.4 rule: {entry['rule_bar']['flips']}"
                                                  if "rule_bar" in entry else ""), flush=True)

    from pruneagent.cli import _run_record

    record = {"task": "1.12", **_run_record(args.model, revision, device, args.dtype, args.seqlen),
              "validation_sha256": manifest["entries"][val_name]["sha256"], "n_windows": n_total,
              "subset_rule": "window i * n_total // n for i < n (evenly spaced, from window 0)",
              "subsets": {str(n): idx for n, idx in subsets.items() if n < n_total},
              "scores": scores.meta, "scores_seconds": scores_seconds,
              "comparison": report, "plans": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
