"""Measured per-layer damage table (G1): one step of each strategy on each layer alone, measured once before the search.

    scripts/mac_gpu.sh caffeinate -i uv run python scripts/damage_table.py \
        --scores <main checkout>/checkpoints/scores/qwen3-0.6b-base__mps__float16.safetensors \
        --data-dir <main checkout>/data_cache

For each of the 28 layers and each of the 4 strategies, starting from the unpruned model, apply ONE step (§4.2
sizes, `plan.step`) to that layer alone, rebuild the model from the cached scores (`pruner.prune`, never
recomputing scores) and measure perplexity on the gate's 32-passage subset (`subset_perplexity`) of the gate text
(`--text`: C4 validation passages by default, the calibration text's source; `wikitext2` reproduces earlier tables). Also
the unpruned model's perplexity on the same windows and, per entry, the rise over it and the weight memory freed
(`estimate.weight_bytes`). Each result is written to the JSON as soon as it is measured; a restart resumes from it.

After the 112 measurements, the quick tests (layer-local probes) are run on the same machine with the same scores
and compared per strategy (Spearman rank correlation across layers), as is the committed quick-test file
(`--quick-tests`, computed on Colab with the earlier WikiText-calibrated FLAP scores).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from pruneagent import data as d
from pruneagent import estimate as E
from pruneagent import evaluate as ev
from pruneagent import plan as P
from pruneagent import probe as pr
from pruneagent import pruner

ROOT = Path(__file__).resolve().parents[1]


def save(record: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(tmp, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--windows", type=int, default=ev.GATE_WINDOWS)
    ap.add_argument("--text", choices=d.GATE_TEXTS, default="c4", help="The gate's passages (data.gate_text_name).")
    ap.add_argument("--scores", type=Path, required=True, help="Cached scores; never recomputed.")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--quick-tests", type=Path, default=ROOT / "results/probes/qwen3-0.6b-base__cuda__float16.json")
    ap.add_argument("--probe-samples", type=int, default=pr.PROBE_SAMPLES)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    slug = d.slug(args.model)
    suffix = "__c4.json" if args.text == "c4" else ".json"  # realrun.DAMAGE_SUFFIX
    out = args.out or ROOT / f"results/damage_table/{slug}__{args.device}__{args.dtype}{suffix}"
    manifest = d.read_manifest(args.data_dir)
    calib = d.calib_name(args.model)
    revision = manifest["entries"][calib]["tokenizer"]["revision"]
    scores = pruner.load_scores(args.scores)
    if scores.meta.get("calib_sha256") != manifest["entries"][calib]["sha256"]:
        raise SystemExit(f"{args.scores} was not computed on {calib}; refusing to use it (scores are never "
                         "recomputed here)")
    val = d.gate_text_name(args.text, args.model)
    ids = d.gate_tokens(args.text, args.model, args.data_dir, revision)
    manifest = d.read_manifest(args.data_dir)  # again: gate_tokens may have just added the C4 passages
    n_total = ids.numel() // args.seqlen

    net = ev.load_model(args.model, revision, args.dtype, args.device)
    cfg = net.config
    dense = P.dense_plan(cfg)
    dense_bytes = E.weight_bytes(cfg, dense)

    want = {"model": args.model, "revision": revision, "device": args.device, "dtype": args.dtype,
            "seqlen": args.seqlen, "windows": args.windows,
            "window_idx": ev.subset_indices(n_total, args.windows),
            "data": val, "data_sha256": manifest["entries"][val]["sha256"],
            "scores_file": args.scores.name, "scores": scores.meta}
    record = json.loads(out.read_text()) if out.exists() else {}
    if record and any(record.get(k) != v for k, v in want.items()):
        raise SystemExit(f"{out} was measured under other settings; move it aside to start again")
    if not record:
        import platform
        from importlib.metadata import version

        record = {"method": "damage_table", **want, "machine": platform.platform(), "torch": version("torch"),
                  "transformers": version("transformers"), "dense": None, "entries": {}, "gpu_seconds": 0.0}
    print(f"{args.model}@{revision[:8]} on {args.device} in {args.dtype}; {args.windows} of {n_total} windows; "
          f"{len(record['entries'])} entries already in {out}", flush=True)

    if record["dense"] is None:
        r = ev.subset_perplexity(net, ids, args.windows, args.seqlen, args.device)
        record["dense"] = {"ppl": r.ppl, "weight_bytes": dense_bytes, "seconds": round(r.seconds, 1)}
        record["gpu_seconds"] += r.seconds
        save(record, out)
    base = record["dense"]["ppl"]
    print(f"  unpruned: ppl {base:.4f}", flush=True)

    todo = [(i, s) for i in range(cfg.num_hidden_layers) for s in P.STRATEGIES
            if f"{i}/{s}" not in record["entries"]]
    for n, (i, s) in enumerate(todo, 1):
        start = time.perf_counter()
        plan = P.step(dense, i, s, cfg)
        pruned = pruner.prune(net, plan, scores)
        r = ev.subset_perplexity(pruned, ids, args.windows, args.seqlen, args.device)
        del pruned
        seconds = time.perf_counter() - start
        freed = dense_bytes - E.weight_bytes(cfg, plan)
        record["entries"][f"{i}/{s}"] = {
            "layer": i, "strategy": s, "setting": dict(plan[i]), "ppl": r.ppl, "increase": r.ppl - base,
            "increase_pct": 100 * (r.ppl / base - 1), "weight_bytes_freed": freed, "seconds": round(seconds, 1)}
        record["gpu_seconds"] += seconds
        save(record, out)
        eta = seconds * (len(todo) - n)
        print(f"  [{n}/{len(todo)}] layer {i:2d} {s:24s} ppl {r.ppl:12.4f} (+{100 * (r.ppl / base - 1):8.2f}%) "
              f"freed {freed / 1e6:6.1f} MB  {seconds:.0f} s; ~{eta / 60:.0f} min left", flush=True)

    # quick tests with the same scores on this machine, then rank agreement
    if "quick_tests" not in record:
        start = time.perf_counter()
        acts = pr.capture(net, d.load_tokens(calib, args.data_dir)[: args.probe_samples], args.device)
        table = pr.sensitivity_table(net, acts, scores)
        ev.synchronize(args.device)
        del acts
        seconds = time.perf_counter() - start
        record["quick_tests"] = {"samples": args.probe_samples, "calib": calib, "seconds": round(seconds, 1),
                                 "table": {str(i): lp for i, lp in table.items()}}
        record["gpu_seconds"] += seconds
    ppl = {(e["layer"], e["strategy"]): e["ppl"] for e in record["entries"].values()}
    tables = {"same_scores": {int(k): v for k, v in record["quick_tests"]["table"].items()}}
    if args.quick_tests.exists():
        tables["committed_file"] = {int(k): v for k, v in json.loads(args.quick_tests.read_text())["table"].items()}
    record["comparison"] = {name: pr.compare(t, base, ppl) for name, t in tables.items()}
    record["comparison_files"] = {"same_scores": "quick_tests (this file)",
                                  "committed_file": str(args.quick_tests.relative_to(ROOT))
                                  if args.quick_tests.is_relative_to(ROOT) else str(args.quick_tests)}
    save(record, out)
    for name, c in record["comparison"].items():
        print(f"  quick tests ({name}) vs measured, Spearman per strategy: "
              + ", ".join(f"{s} {v['spearman']:.2f}" for s, v in c["per_strategy"].items()), flush=True)
    print(f"Saved {out}; Mac GPU time {record['gpu_seconds'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
