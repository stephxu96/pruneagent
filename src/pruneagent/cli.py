"""Command-line entry point. Further subcommands (run, report) arrive in later tasks."""

from pathlib import Path
from typing import Annotated

import typer

from pruneagent import __version__

app = typer.Typer(help="PruneAgent: prune Qwen3 to Wanda-level quality with real savings.", no_args_is_help=True)


@app.command()
def version() -> None:
    """Print the PruneAgent version."""
    typer.echo(__version__)


@app.command()
def data(
    model: Annotated[str, typer.Option(help="Model whose tokenizer is used.")] = "Qwen/Qwen3-0.6B-Base",
    nsamples: Annotated[int, typer.Option(help="Calibration samples.")] = 128,
    seqlen: Annotated[int, typer.Option(help="Tokens per calibration sample and evaluation window.")] = 2048,
    seed: Annotated[int, typer.Option(help="Calibration sampling seed (0 matches the Wanda reference).")] = 0,
    out: Annotated[
        Path | None, typer.Option(help="Cache directory (default: $PRUNEAGENT_CACHE or ./data_cache).")
    ] = None,
    check: Annotated[bool, typer.Option("--verify", help="Only check cached files against the manifest.")] = False,
) -> None:
    """Download, tokenize and cache C4 calibration data and WikiText-2 validation and test splits."""
    from pruneagent import data as d

    if check:
        result = d.verify(out)
        if not result:
            typer.echo("Cache is empty.")
            raise typer.Exit(1)
        for name, ok in result.items():
            typer.echo(f"{'ok  ' if ok else 'FAIL'} {name}")
        raise typer.Exit(0 if all(result.values()) else 1)

    manifest = d.build_cache(model, nsamples, seqlen, seed, out, log=typer.echo)
    typer.echo(f"Manifest: {(out or d.cache_dir()) / d.MANIFEST} ({len(manifest['entries'])} entries)")


@app.command()
def ppl(
    model: Annotated[str, typer.Option(help="Hugging Face model id.")] = "Qwen/Qwen3-0.6B-Base",
    dtype: Annotated[str, typer.Option(help="float16, bfloat16 or float32.")] = "float16",
    splits: Annotated[list[str] | None, typer.Option("--split", help="WikiText-2 split(s).")] = None,
    seqlen: Annotated[int, typer.Option(help="Window length in tokens.")] = 2048,
    device: Annotated[str | None, typer.Option(help="cuda, mps or cpu (default: best available).")] = None,
    data_dir: Annotated[Path | None, typer.Option(help="Data cache (default: $PRUNEAGENT_CACHE or ./data_cache).")] = None,
    results: Annotated[Path, typer.Option(help="Directory for the results JSON.")] = Path("results/dense"),
) -> None:
    """WikiText-2 perplexity of an unpruned model, Wanda-style, on cached tokens. Writes a results JSON."""
    import json
    import platform
    from datetime import UTC, datetime
    from importlib.metadata import version as pkg_version

    from pruneagent import data as d
    from pruneagent import evaluate as ev

    splits = splits or ["validation", "test"]
    device = device or ev.pick_device()
    manifest = d.read_manifest(data_dir)
    tokenizer_rev = manifest["entries"][d.wikitext_name(splits[0], model)]["tokenizer"]["revision"]
    typer.echo(f"{model}@{tokenizer_rev[:8]} on {device} in {dtype}")
    net = ev.load_model(model, tokenizer_rev, dtype, device)

    record = {
        "model": model, "revision": tokenizer_rev, "device": device, "dtype": dtype, "seqlen": seqlen,
        "machine": platform.platform(), "processor": platform.processor() or platform.machine(),
        "torch": pkg_version("torch"), "transformers": pkg_version("transformers"),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"), "splits": {},
    }
    for split in splits:
        name = d.wikitext_name(split, model)
        r = ev.perplexity(net, d.load_tokens(name, data_dir), seqlen, device)
        record["splits"][split] = {
            "ppl": r.ppl, "n_windows": r.n_windows, "seconds": round(r.seconds, 1),
            "data_sha256": manifest["entries"][name]["sha256"],
        }
        typer.echo(f"  {split:<10} ppl {r.ppl:.4f}  ({r.n_windows} windows, {r.seconds:.0f} s)")

    results.mkdir(parents=True, exist_ok=True)
    out = results / f"{d.slug(model)}__{device}__{dtype}.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    typer.echo(f"Saved {out}")


@app.command("wanda")
def wanda_cmd(
    model: Annotated[str, typer.Option(help="Hugging Face model id.")] = "Qwen/Qwen3-0.6B-Base",
    sparsity: Annotated[float, typer.Option(help="Unstructured sparsity in every block's linear layers.")] = 0.5,
    dtype: Annotated[str, typer.Option(help="float16, bfloat16 or float32.")] = "float16",
    seqlen: Annotated[int, typer.Option(help="Evaluation window length.")] = 2048,
    device: Annotated[str | None, typer.Option(help="cuda, mps or cpu (default: best available).")] = None,
    data_dir: Annotated[Path | None, typer.Option(help="Data cache (default: $PRUNEAGENT_CACHE or ./data_cache).")] = None,
    results: Annotated[Path, typer.Option(help="Directory for the results JSON.")] = Path("results/wanda"),
) -> None:
    """Prune with Wanda (all block linear layers, C4 calibration, layer by layer) and report WikiText-2 perplexity."""
    import json
    import platform
    import time
    from datetime import UTC, datetime
    from importlib.metadata import version as pkg_version

    from pruneagent import data as d
    from pruneagent import evaluate as ev
    from pruneagent import wanda

    device = device or ev.pick_device()
    manifest = d.read_manifest(data_dir)
    calib_entry = manifest["entries"][d.calib_name(model)]
    revision = calib_entry["tokenizer"]["revision"]
    typer.echo(f"{model}@{revision[:8]} on {device} in {dtype}: Wanda at {sparsity:.0%}")
    net = ev.load_model(model, revision, dtype, device)
    calib = d.load_tokens(d.calib_name(model), data_dir)

    start = time.perf_counter()
    wanda.prune_wanda(net, calib, sparsity, device)
    ev.synchronize(device)
    prune_seconds = time.perf_counter() - start
    achieved = wanda.linear_sparsity(net)
    typer.echo(f"  pruned in {prune_seconds:.0f} s; block linear sparsity {achieved:.4f}")

    record = {
        "method": "wanda", "target_sparsity": sparsity, "achieved_linear_sparsity": achieved,
        "model": model, "revision": revision, "device": device, "dtype": dtype, "seqlen": seqlen,
        "calibration": {"entry": d.calib_name(model), "sha256": calib_entry["sha256"], "shape": calib_entry["shape"]},
        "prune_seconds": round(prune_seconds, 1), "machine": platform.platform(),
        "torch": pkg_version("torch"), "transformers": pkg_version("transformers"),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"), "splits": {},
    }
    for split in ("validation", "test"):
        name = d.wikitext_name(split, model)
        r = ev.perplexity(net, d.load_tokens(name, data_dir), seqlen, device)
        record["splits"][split] = {"ppl": r.ppl, "n_windows": r.n_windows, "seconds": round(r.seconds, 1),
                                   "data_sha256": manifest["entries"][name]["sha256"]}
        typer.echo(f"  {split:<10} ppl {r.ppl:.4f}  ({r.n_windows} windows, {r.seconds:.0f} s)")

    results.mkdir(parents=True, exist_ok=True)
    out = results / f"{d.slug(model)}__wanda{round(sparsity * 100)}__{device}__{dtype}.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    typer.echo(f"Saved {out}")


@app.command("flap")
def flap_cmd(
    model: Annotated[str, typer.Option(help="Hugging Face model id.")] = "Qwen/Qwen3-0.6B-Base",
    ratio: Annotated[float, typer.Option(help="Share of block weights FLAP removes.")] = 0.2,
    dtype: Annotated[str, typer.Option(help="float16, bfloat16 or float32.")] = "float16",
    seqlen: Annotated[int, typer.Option(help="Evaluation window length.")] = 2048,
    device: Annotated[str | None, typer.Option(help="cuda, mps or cpu (default: best available).")] = None,
    data_dir: Annotated[Path | None, typer.Option(help="Data cache (default: $PRUNEAGENT_CACHE or ./data_cache).")] = None,
    results: Annotated[Path, typer.Option(help="Directory for the results JSON.")] = Path("results/flap"),
) -> None:
    """Prune with FLAP at its own allocation (WIFV, AL-AM, bias compensation) and report WikiText-2 perplexity."""
    import json
    import platform
    import time
    from datetime import UTC, datetime
    from importlib.metadata import version as pkg_version

    from pruneagent import data as d
    from pruneagent import evaluate as ev
    from pruneagent import flap

    device = device or ev.pick_device()
    manifest = d.read_manifest(data_dir)
    calib_entry = manifest["entries"][d.flap_calib_name(model)]
    revision = calib_entry["tokenizer"]["revision"]
    typer.echo(f"{model}@{revision[:8]} on {device} in {dtype}: FLAP at {ratio:.0%}")
    net = ev.load_model(model, revision, dtype, device)
    before = flap.block_params(net)

    start = time.perf_counter()
    out = flap.prune_flap(net, d.load_tokens(d.flap_calib_name(model), data_dir), ratio, device)
    ev.synchronize(device)
    prune_seconds = time.perf_counter() - start
    removed = 1 - flap.block_params(net) / before
    groups = out["attn_keep"].sum(dim=1).tolist()
    neurons = out["mlp_keep"].sum(dim=1).tolist()
    typer.echo(f"  pruned in {prune_seconds:.0f} s; block weights removed {removed:.4f}")
    typer.echo(f"  groups kept per layer  {groups}")
    typer.echo(f"  neurons kept per layer {neurons}")

    record = {
        "method": "flap (WIFV, AL-AM, bias compensation)", "target_ratio": ratio, "block_weights_removed": removed,
        "groups_kept": groups, "neurons_kept": neurons,
        "model": model, "revision": revision, "device": device, "dtype": dtype, "seqlen": seqlen,
        "calibration": {"entry": d.flap_calib_name(model), "sha256": calib_entry["sha256"],
                        "shape": calib_entry["shape"]},
        "prune_seconds": round(prune_seconds, 1), "machine": platform.platform(),
        "torch": pkg_version("torch"), "transformers": pkg_version("transformers"),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"), "splits": {},
    }
    for split in ("validation", "test"):
        name = d.wikitext_name(split, model)
        r = ev.perplexity(net, d.load_tokens(name, data_dir), seqlen, device)
        record["splits"][split] = {"ppl": r.ppl, "n_windows": r.n_windows, "seconds": round(r.seconds, 1),
                                   "data_sha256": manifest["entries"][name]["sha256"]}
        typer.echo(f"  {split:<10} ppl {r.ppl:.4f}  ({r.n_windows} windows, {r.seconds:.0f} s)")

    results.mkdir(parents=True, exist_ok=True)
    out_path = results / f"{d.slug(model)}__flap{round(ratio * 100)}__{device}__{dtype}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    typer.echo(f"Saved {out_path}")


@app.command("wanda-sp")
def wanda_sp_cmd(
    model: Annotated[str, typer.Option(help="Hugging Face model id.")] = "Qwen/Qwen3-0.6B-Base",
    ratio: Annotated[float, typer.Option(help="Share of attention groups and MLP neurons removed in every block.")] = 0.2,
    dtype: Annotated[str, typer.Option(help="float16, bfloat16 or float32.")] = "float16",
    seqlen: Annotated[int, typer.Option(help="Evaluation window length.")] = 2048,
    device: Annotated[str | None, typer.Option(help="cuda, mps or cpu (default: best available).")] = None,
    data_dir: Annotated[Path | None, typer.Option(help="Data cache (default: $PRUNEAGENT_CACHE or ./data_cache).")] = None,
    results: Annotated[Path, typer.Option(help="Directory for the results JSON.")] = Path("results/wanda_sp"),
) -> None:
    """Prune with uniform Wanda-sp (same share in every block, no bias compensation) and report WikiText-2 perplexity."""
    import json
    import platform
    import time
    from datetime import UTC, datetime
    from importlib.metadata import version as pkg_version

    from pruneagent import data as d
    from pruneagent import evaluate as ev
    from pruneagent import flap, wanda_sp

    device = device or ev.pick_device()
    manifest = d.read_manifest(data_dir)
    calib_entry = manifest["entries"][d.wanda_sp_calib_name(model)]
    revision = calib_entry["tokenizer"]["revision"]
    typer.echo(f"{model}@{revision[:8]} on {device} in {dtype}: Wanda-sp at {ratio:.0%}")
    net = ev.load_model(model, revision, dtype, device)
    before = flap.block_params(net)

    start = time.perf_counter()
    out = wanda_sp.prune_wanda_sp(net, d.load_tokens(d.wanda_sp_calib_name(model), data_dir), ratio, device)
    ev.synchronize(device)
    prune_seconds = time.perf_counter() - start
    removed = 1 - flap.block_params(net) / before
    groups = out["attn_keep"].sum(dim=1).tolist()
    neurons = out["mlp_keep"].sum(dim=1).tolist()
    typer.echo(f"  pruned in {prune_seconds:.0f} s; block weights removed {removed:.4f}")
    typer.echo(f"  groups kept per layer  {groups}")
    typer.echo(f"  neurons kept per layer {neurons}")

    record = {
        "method": "wanda-sp (uniform, no bias compensation)", "target_ratio": ratio, "block_weights_removed": removed,
        "groups_kept": groups, "neurons_kept": neurons,
        "model": model, "revision": revision, "device": device, "dtype": dtype, "seqlen": seqlen,
        "calibration": {"entry": d.wanda_sp_calib_name(model), "sha256": calib_entry["sha256"],
                        "shape": calib_entry["shape"]},
        "prune_seconds": round(prune_seconds, 1), "machine": platform.platform(),
        "torch": pkg_version("torch"), "transformers": pkg_version("transformers"),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"), "splits": {},
    }
    for split in ("validation", "test"):
        name = d.wikitext_name(split, model)
        r = ev.perplexity(net, d.load_tokens(name, data_dir), seqlen, device)
        record["splits"][split] = {"ppl": r.ppl, "n_windows": r.n_windows, "seconds": round(r.seconds, 1),
                                   "data_sha256": manifest["entries"][name]["sha256"]}
        typer.echo(f"  {split:<10} ppl {r.ppl:.4f}  ({r.n_windows} windows, {r.seconds:.0f} s)")

    results.mkdir(parents=True, exist_ok=True)
    out_path = results / f"{d.slug(model)}__wandasp{round(ratio * 100)}__{device}__{dtype}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    typer.echo(f"Saved {out_path}")


@app.command("mc")
def mc_cmd(
    model: Annotated[str, typer.Option(help="Hugging Face model id.")] = "Qwen/Qwen3-0.6B-Base",
    sparsity: Annotated[float, typer.Option(help="Wanda sparsity to apply first (0 = unpruned).")] = 0.0,
    dtype: Annotated[str, typer.Option(help="float16, bfloat16 or float32.")] = "float16",
    batch_size: Annotated[str, typer.Option(help="lm-eval batch size, or 'auto'.")] = "auto",
    limit: Annotated[int | None, typer.Option(help="Questions per task (for quick checks only).")] = None,
    device: Annotated[str | None, typer.Option(help="cuda, mps or cpu (default: best available).")] = None,
    data_dir: Annotated[Path | None, typer.Option(help="Data cache (default: $PRUNEAGENT_CACHE or ./data_cache).")] = None,
    results: Annotated[Path, typer.Option(help="Directory for the results JSON.")] = Path("results/mc"),
) -> None:
    """Multiple-choice accuracy on the seven Wanda-paper question sets, optionally after Wanda pruning."""
    import json
    import platform
    import time
    from datetime import UTC, datetime
    from importlib.metadata import version as pkg_version

    from transformers import AutoTokenizer

    from pruneagent import data as d
    from pruneagent import evaluate as ev
    from pruneagent import mc, wanda

    device = device or ev.pick_device()
    manifest = d.read_manifest(data_dir)
    revision = manifest["entries"][d.calib_name(model)]["tokenizer"]["revision"]
    label = f"wanda{round(sparsity * 100)}" if sparsity else "dense"
    typer.echo(f"{model}@{revision[:8]} on {device} in {dtype}: {label}")
    net = ev.load_model(model, revision, dtype, device)
    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    record = {"method": label, "model": model, "revision": revision, "device": device, "dtype": dtype}
    if sparsity:
        wanda.prune_wanda(net, d.load_tokens(d.calib_name(model), data_dir), sparsity, device)
        record["achieved_linear_sparsity"] = wanda.linear_sparsity(net)
        typer.echo(f"  block linear sparsity {record['achieved_linear_sparsity']:.4f}")

    start = time.perf_counter()
    raw = mc.evaluate(net, tok, device, batch_size=int(batch_size) if batch_size.isdigit() else batch_size,
                      limit=limit)
    record |= mc.summarize(raw)
    record |= {
        "num_fewshot": 0, "limit": limit, "seconds": round(time.perf_counter() - start, 1),
        "machine": platform.platform(), "torch": pkg_version("torch"), "transformers": pkg_version("transformers"),
        "lm_eval": pkg_version("lm_eval"), "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    for task, r in record["tasks"].items():
        typer.echo(f"  {task:<14} acc {r['acc']:.4f}")
    typer.echo(f"  {'mean':<14} acc {record['mean_acc']:.4f}  ({record['seconds']:.0f} s)")

    results.mkdir(parents=True, exist_ok=True)
    suffix = f"__limit{limit}" if limit else ""
    out = results / f"{d.slug(model)}__{label}__{device}__{dtype}{suffix}.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    typer.echo(f"Saved {out}")


def _run_record(model: str, revision: str, device: str, dtype: str, seqlen: int) -> dict:
    import platform
    from datetime import UTC, datetime
    from importlib.metadata import version as pkg_version

    return {"model": model, "revision": revision, "device": device, "dtype": dtype, "seqlen": seqlen,
            "machine": platform.platform(), "torch": pkg_version("torch"),
            "transformers": pkg_version("transformers"),
            "created_at": datetime.now(UTC).isoformat(timespec="seconds")}


def _val_ppl(net, model: str, seqlen: int, windows: int | None, device: str, data_dir: Path | None) -> dict:
    from pruneagent import data as d
    from pruneagent import evaluate as ev

    name = d.wikitext_name("validation", model)
    ids = d.load_tokens(name, data_dir).reshape(-1)
    if windows:
        ids = ids[: windows * seqlen]
    r = ev.perplexity(net, ids, seqlen, device)
    sha = d.read_manifest(data_dir)["entries"][name]["sha256"]
    return {"ppl": r.ppl, "n_windows": r.n_windows, "seconds": round(r.seconds, 1), "data_sha256": sha}


def _scores(net, model: str, revision: str, device: str, dtype: str, data_dir: Path | None,
            scores_path: Path | None):
    """Cached pruning scores for the unpruned `net`, computed and saved if missing."""
    import time

    from pruneagent import data as d
    from pruneagent import pruner

    scores_path = scores_path or Path("checkpoints/scores") / f"{d.slug(model)}__{device}__{dtype}.safetensors"
    manifest = d.read_manifest(data_dir)
    calib = d.calib_name(model)
    want = {"calib": calib, "calib_sha256": manifest["entries"][calib]["sha256"]}
    if scores_path.exists():
        cached = pruner.load_scores(scores_path)
        if all(cached.meta.get(k) == v for k, v in want.items()):
            typer.echo(f"  scores from {scores_path}")
            return cached
        typer.echo(f"  {scores_path} was computed on other calibration data "
                   f"({cached.meta.get('flap_calib') or cached.meta.get('calib')}); recomputing on {calib}")
    start = time.perf_counter()
    meta = {"model": model, "revision": revision, "device": device, "dtype": dtype, **want}
    scores = pruner.compute_scores(net, d.load_tokens(calib, data_dir), device, meta=meta)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    pruner.save_scores(scores, scores_path)
    typer.echo(f"  scores computed in {time.perf_counter() - start:.0f} s -> {scores_path}")
    return scores


@app.command("prune")
def prune_cmd(
    plan: Annotated[Path, typer.Option(help="Plan JSON: {'plan': {layer: {...}}} and/or {'steps': [...]}.")],
    out: Annotated[Path, typer.Option(help="Directory for model.safetensors and plan.json.")],
    model: Annotated[str, typer.Option(help="Hugging Face model id.")] = "Qwen/Qwen3-0.6B-Base",
    dtype: Annotated[str, typer.Option(help="float16, bfloat16 or float32.")] = "float16",
    seqlen: Annotated[int, typer.Option(help="Evaluation window length.")] = 2048,
    windows: Annotated[int | None, typer.Option(help="Validation windows to evaluate (default: all).")] = None,
    reload_check: Annotated[bool, typer.Option(help="Reload from disk and evaluate again.")] = False,
    scores_path: Annotated[Path | None, typer.Option("--scores", help="Cached scores (computed if missing).")] = None,
    device: Annotated[str | None, typer.Option(help="cuda, mps or cpu (default: best available).")] = None,
    data_dir: Annotated[Path | None, typer.Option(help="Data cache (default: $PRUNEAGENT_CACHE or ./data_cache).")] = None,
    results: Annotated[Path, typer.Option(help="Directory for the results JSON.")] = Path("results/pruned"),
) -> None:
    """Prune the unpruned model to a per-layer plan, save it, and report WikiText-2 validation perplexity."""
    import gc
    import json
    import time

    from pruneagent import data as d
    from pruneagent import evaluate as ev
    from pruneagent import loader, pruner
    from pruneagent import plan as P

    device = device or ev.pick_device()
    manifest = d.read_manifest(data_dir)
    revision = manifest["entries"][d.calib_name(model)]["tokenizer"]["revision"]
    spec = json.loads(plan.read_text())
    typer.echo(f"{model}@{revision[:8]} on {device} in {dtype}: plan {plan}")
    net = ev.load_model(model, revision, dtype, device)
    layer_plan = P.apply_steps(P.validate(spec.get("plan", {}), net.config), spec.get("steps", []), net.config)

    scores = _scores(net, model, revision, device, dtype, data_dir, scores_path)

    before = pruner.weight_counts(net)
    start = time.perf_counter()
    pruner.apply_plan(net, layer_plan, scores)
    ev.synchronize(device)
    prune_seconds = time.perf_counter() - start
    after = pruner.weight_counts(net)
    loader.save(net, out, model_id=model, revision=revision, extra={"scores": scores.meta})
    removed = 1 - after["block_weights"] / before["block_weights"]
    typer.echo(f"  pruned in {prune_seconds:.1f} s; block weights removed {removed:.4f}; "
               f"zeros {after['block_zeros'] / after['block_weights']:.4f} of the rest; saved {out}")

    record = {"method": "plan", **_run_record(model, revision, device, dtype, seqlen),
              "plan_file": str(plan), "saved_to": str(out), "steps": spec.get("steps", []),
              "plan": P.to_json(layer_plan),
              "dropped_layers": [i for i, p in layer_plan.items() if p["dropped"]],
              "weights_before": before, "weights_after": after, "block_weights_removed": removed,
              "parameters_removed": 1 - after["parameters"] / before["parameters"],
              "scores": scores.meta, "prune_seconds": round(prune_seconds, 2)}
    record["validation"] = _val_ppl(net, model, seqlen, windows, device, data_dir)
    typer.echo(f"  validation ppl {record['validation']['ppl']:.6f} ({record['validation']['n_windows']} windows)")
    if reload_check:
        del net
        gc.collect()
        back = loader.load(out, device=device)
        record["validation_after_reload"] = _val_ppl(back, model, seqlen, windows, device, data_dir)
        record["identical_after_reload"] = record["validation_after_reload"]["ppl"] == record["validation"]["ppl"]
        typer.echo(f"  after reload   {record['validation_after_reload']['ppl']:.6f}  "
                   f"identical: {record['identical_after_reload']}")

    results.mkdir(parents=True, exist_ok=True)
    path = results / f"{d.slug(model)}__{plan.stem}__{device}__{dtype}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    typer.echo(f"Saved {path}")


@app.command("ppl-saved")
def ppl_saved_cmd(
    path: Annotated[Path, typer.Option(help="Saved pruned model directory (model.safetensors + plan.json).")],
    seqlen: Annotated[int, typer.Option(help="Evaluation window length.")] = 2048,
    windows: Annotated[int | None, typer.Option(help="Validation windows to evaluate (default: all).")] = None,
    device: Annotated[str | None, typer.Option(help="cuda, mps or cpu (default: best available).")] = None,
    data_dir: Annotated[Path | None, typer.Option(help="Data cache (default: $PRUNEAGENT_CACHE or ./data_cache).")] = None,
    results: Annotated[Path, typer.Option(help="Directory for the results JSON.")] = Path("results/pruned"),
) -> None:
    """WikiText-2 validation perplexity of a saved pruned model, rebuilt from its plan JSON."""
    import json

    from pruneagent import evaluate as ev
    from pruneagent import loader

    device = device or ev.pick_device()
    net = loader.load(path, device=device)
    info = net.pruneagent
    dtype = str(next(net.parameters()).dtype).removeprefix("torch.")
    typer.echo(f"{path}: {info['model_id']}@{(info['revision'] or '')[:8]} on {device} in {dtype}")
    record = {"method": "saved plan", **_run_record(info["model_id"], info["revision"], device, dtype, seqlen),
              "path": str(path), "weights_sha256": loader.read_plan(path)["weights"]["sha256"],
              "validation": _val_ppl(net, info["model_id"], seqlen, windows, device, data_dir)}
    typer.echo(f"  validation ppl {record['validation']['ppl']:.6f} ({record['validation']['n_windows']} windows)")
    results.mkdir(parents=True, exist_ok=True)
    out = results / f"{path.name}__saved__{device}__{dtype}.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    typer.echo(f"Saved {out}")


WANDA50_REF_PPL = Path("results/wanda/qwen3-0.6b-base__wanda50__mps__float16.json")
WANDA50_REF_MC = Path("results/mc/qwen3-0.6b-base__colab.json")
WANDA50_REF_MC_RUN = "qwen3-0.6b-base__wanda50__cuda__float16.json"


@app.command("compare")
def compare_cmd(
    methods: Annotated[list[str] | None, typer.Option("--method", help="flap and/or wanda-sp (default: both).")] = None,
    model: Annotated[str, typer.Option(help="Hugging Face model id.")] = "Qwen/Qwen3-0.6B-Base",
    dtype: Annotated[str, typer.Option(help="float16, bfloat16 or float32.")] = "float16",
    lo: Annotated[float, typer.Option(help="Ratio assumed to keep Wanda-50% quality (not evaluated).")] = 0.0,
    hi: Annotated[float, typer.Option(help="Ratio expected to fall short of Wanda-50% (evaluated first).")] = 0.5,
    tol: Annotated[float, typer.Option(help="Stop bisecting when the ratio interval is this narrow.")] = 0.005,
    acc_lo: Annotated[float, typer.Option(help="Lowest ratio the accuracy bisection considers.")] = 0.0,
    acc_tol: Annotated[float, typer.Option(help="Accuracy bisection stops at this interval width.")] = 0.01,
    acc_fails_at: Annotated[float | None, typer.Option(help="Ratio already known to fail accuracy.")] = None,
    ppl_margin: Annotated[float, typer.Option(help="Allowed factor over Wanda-50% validation perplexity "
                                                   "(default 1.0: no allowance; task 1.9 used 1.10).")] = 1.0,
    acc_margin: Annotated[float, typer.Option(help="Allowed drop below Wanda-50% mean accuracy "
                                                   "(default 0: no allowance; task 1.9 used 0.02).")] = 0.0,
    max_ppl: Annotated[float | None, typer.Option(help="Validation perplexity limit; overrides --ppl-margin.")] = None,
    min_acc: Annotated[float | None, typer.Option(help="Mean accuracy floor; overrides --acc-margin.")] = None,
    seqlen: Annotated[int, typer.Option(help="Evaluation window length.")] = 2048,
    windows: Annotated[int | None, typer.Option(help="Validation windows (default: all; fewer for quick checks).")] = None,
    mc_limit: Annotated[int | None, typer.Option(help="Questions per task (quick checks only).")] = None,
    device: Annotated[str | None, typer.Option(help="cuda, mps or cpu (default: best available).")] = None,
    data_dir: Annotated[Path | None, typer.Option(help="Data cache (default: $PRUNEAGENT_CACHE or ./data_cache).")] = None,
    results: Annotated[Path, typer.Option(help="Directory for the results JSON.")] = Path("results/baselines"),
) -> None:
    """FLAP and uniform Wanda-sp pruned to the largest ratio that keeps the quality of the Wanda-50% reference
    (a backup comparison since v0.8; no allowance by default): bisect the ratio on validation perplexity, then
    check accuracy there and bisect lower on both if needed. Records the savings at that ratio.
    Baseline: FLAP at a memory limit is `baselines.flap_at_memory_limit`, not yet wired to this command."""
    import copy
    import gc
    import json
    import time

    import torch
    from transformers import AutoTokenizer

    from pruneagent import baselines as B
    from pruneagent import data as d
    from pruneagent import evaluate as ev
    from pruneagent import flap, mc, wanda_sp

    methods = methods or ["flap", "wanda-sp"]
    if unknown := set(methods) - {"flap", "wanda-sp"}:
        raise typer.BadParameter(f"unknown method(s): {sorted(unknown)}")
    device = device or ev.pick_device()
    manifest = d.read_manifest(data_dir)
    revision = manifest["entries"][d.calib_name(model)]["tokenizer"]["revision"]
    base = B.wanda50_reference(WANDA50_REF_PPL, WANDA50_REF_MC, WANDA50_REF_MC_RUN, ppl_margin, acc_margin)
    base["max_validation_ppl"] = max_ppl if max_ppl is not None else base["max_validation_ppl"]
    base["min_mean_acc"] = min_acc if min_acc is not None else base["min_mean_acc"]
    typer.echo(f"{model}@{revision[:8]} on {device} in {dtype}; Wanda-50% quality: validation ppl <= "
               f"{base['max_validation_ppl']:.4f}, mean accuracy >= {base['min_mean_acc']:.4f}")
    dense = ev.load_model(model, revision, dtype, device)
    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    before = {**B.memory(dense), "block_weights": flap.block_params(dense)}
    val_ids = d.load_tokens(d.wikitext_name("validation", model), data_dir).reshape(-1)
    if windows:
        val_ids = val_ids[: windows * seqlen]
    test_ids = d.load_tokens(d.wikitext_name("test", model), data_dir)

    for method in methods:
        if method == "flap":
            calib_name = d.flap_calib_name(model)
            start = time.perf_counter()
            stats = flap.collect(dense, d.load_tokens(calib_name, data_dir), device)
            typer.echo(f"[flap] statistics collected once in {time.perf_counter() - start:.0f} s")

            def prune_fn(net, ratio, stats=stats):
                return flap.prune_flap_from_stats(net, stats, ratio)
        else:
            calib_name = d.wanda_sp_calib_name(model)
            calib = d.load_tokens(calib_name, data_dir)

            def prune_fn(net, ratio, calib=calib):
                return wanda_sp.prune_wanda_sp(net, calib, ratio, device)
        current: dict = {"ratio": None, "net": None}
        info: dict[float, dict] = {}
        record = {"method": method, **_run_record(model, revision, device, dtype, seqlen), "wanda50_reference": base,
                  "calibration": {"entry": calib_name, "sha256": manifest["entries"][calib_name]["sha256"]},
                  "search": {"lo": lo, "hi": hi, "tol": tol, "acc_lo": acc_lo, "acc_tol": acc_tol,
                             "acc_fails_at": acc_fails_at, "windows": windows, "mc_limit": mc_limit,
                             "status": "running"},
                  "weights_before": before}
        suffix = f"__windows{windows}" if windows else ""
        suffix += f"__limit{mc_limit}" if mc_limit else ""
        path = results / f"{d.slug(model)}__{method.replace('-', '')}-at-wanda50__{device}__{dtype}{suffix}.json"

        def save(record=record, info=info, path=path) -> None:
            """Write everything measured so far; called after every measurement so a crash loses nothing."""
            record["trials"] = {f"{k:.4f}": v for k, v in sorted(info.items())}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=2) + "\n")

        def build(ratio: float, prune_fn=prune_fn, current=current, info=info) -> object:
            if current["ratio"] == ratio:
                return current["net"]
            current["net"] = None
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            net = copy.deepcopy(dense)
            start = time.perf_counter()
            out = prune_fn(net, ratio)
            ev.synchronize(device)
            after = {**B.memory(net), "block_weights": flap.block_params(net)}
            entry = info.setdefault(ratio, {"builds": 0})  # a rebuild keeps earlier measurements at this ratio
            entry["builds"] += 1
            entry |= {
                "prune_seconds": round(time.perf_counter() - start, 1), "weights_after": after,
                "block_weights_removed": 1 - after["block_weights"] / before["block_weights"],
                "parameters_removed": 1 - after["parameters"] / before["parameters"],
                "weight_bytes_removed": 1 - after["weight_bytes"] / before["weight_bytes"],
                "groups_kept": out["attn_keep"].sum(dim=1).tolist(),
                "neurons_kept": out["mlp_keep"].sum(dim=1).tolist(),
            }
            current.update(ratio=ratio, net=net)
            return net

        def val_ppl(ratio: float, method=method, info=info) -> float:
            net = build(ratio)
            r = ev.perplexity(net, val_ids, seqlen, device)
            info[ratio]["validation"] = {"ppl": r.ppl, "n_windows": r.n_windows, "seconds": round(r.seconds, 1)}
            typer.echo(f"[{method}] ratio {ratio:.4f}: block weights removed {info[ratio]['block_weights_removed']:.4f}"
                       f", validation ppl {r.ppl:.4f}")
            save()
            return r.ppl

        def mean_acc(ratio: float, method=method, info=info) -> float:
            net = build(ratio)
            start = time.perf_counter()
            raw = mc.evaluate(net, tok, device, batch_size="auto", limit=mc_limit)
            info[ratio]["mc"] = mc.summarize(raw) | {"seconds": round(time.perf_counter() - start, 1)}
            typer.echo(f"[{method}] ratio {ratio:.4f}: mean accuracy {info[ratio]['mc']['mean_acc']:.4f}; " + ", ".join(
                f"{t} {v['acc']:.4f}" for t, v in info[ratio]["mc"]["tasks"].items()))
            save()
            return info[ratio]["mc"]["mean_acc"]

        search = B.search_wanda50_match(val_ppl, mean_acc, base["max_validation_ppl"], base["min_mean_acc"],
                                   lo=lo, hi=hi, tol=tol, acc_lo=acc_lo, acc_tol=acc_tol, acc_fails_at=acc_fails_at)
        record["search"] |= {"status": "done", **search.to_json()}
        save()
        if search.ratio is not None:
            net = build(search.ratio)  # rebuilt only if a later trial replaced it
            at = info[search.ratio]
            r = ev.perplexity(net, val_ids, seqlen, device)
            at["validation_final_model"] = {"ppl": r.ppl, "n_windows": r.n_windows}
            save()
            if windows is None:
                r = ev.perplexity(net, test_ids, seqlen, device)
                at["test"] = {"ppl": r.ppl, "n_windows": r.n_windows, "seconds": round(r.seconds, 1)}
                typer.echo(f"[{method}] ratio {search.ratio:.4f}: test ppl {r.ppl:.4f}")
                save()
            record["at_wanda50_match"] = {"ratio": search.ratio, **at}
            typer.echo(f"[{method}] keeps Wanda-50% quality up to ratio {search.ratio:.4f}, block weights removed "
                       f"{at['block_weights_removed']:.4f}, weight bytes removed {at['weight_bytes_removed']:.4f}, "
                       f"validation {at.get('validation', {}).get('ppl', float('nan')):.4f}, test {at.get('test', {}).get('ppl', float('nan')):.4f}"
                       f", mean accuracy {at.get('mc', {}).get('mean_acc', float('nan')):.4f}")
        else:
            typer.echo(f"[{method}] Wanda-50% quality not kept: {search.note}")
        current["net"] = None
        save()
        typer.echo(f"Saved {path}")


# Deprecated alias: `baseline` was this command's name until architecture v0.8, when Baseline: FLAP became the
# primary comparison; click prints a deprecation notice when it is used.
app.command("baseline", deprecated=True, hidden=True,
            help="Deprecated alias of `compare`.")(compare_cmd)


@app.command("estimate")
def estimate_cmd(
    plan: Annotated[Path, typer.Option(help="Plan JSON ({'plan': ...} and/or {'steps': [...]}), or a saved model's "
                                            "plan.json or directory (then compared with its file size).")],
    target: Annotated[str | None, typer.Option(help="Device target: preset (m4, t4), JSON file or inline JSON.")] = None,
    model: Annotated[str, typer.Option(help="Hugging Face model id (ignored for a saved model).")] = "Qwen/Qwen3-0.6B-Base",
    dtype: Annotated[str, typer.Option(help="float16, bfloat16 or float32 (ignored for a saved model).")] = "float16",
    context: Annotated[int, typer.Option(help="Tokens of history when generating (key/value cache).")] = 512,
    prompt: Annotated[int, typer.Option(help="Prompt length for prefill FLOPs.")] = 512,
    data_dir: Annotated[Path | None, typer.Option(help="Data cache (default: $PRUNEAGENT_CACHE or ./data_cache).")] = None,
    results: Annotated[Path, typer.Option(help="Directory for the results JSON.")] = Path("results/estimate"),
) -> None:
    """Weight bytes, bytes read per generated token, prompt FLOPs and estimated speed for a plan; no weights needed."""
    import json

    from pruneagent import data as d
    from pruneagent import estimate as E
    from pruneagent import loader
    from pruneagent import plan as P

    path = plan / loader.PLAN if plan.is_dir() else plan
    spec = json.loads(path.read_text())
    is_saved = spec.get("format") == loader.FORMAT
    record: dict = {"plan_file": str(path)}
    if is_saved:
        saved = loader.read_plan(path.parent)
        config, layer_plan, dtype = saved["dense_config"], saved["plan"], saved["dtype"]
        model, revision = saved["model_id"], saved["revision"]
    else:
        from transformers import AutoConfig

        revision = d.read_manifest(data_dir)["entries"][d.calib_name(model)]["tokenizer"]["revision"]
        config = AutoConfig.from_pretrained(model, revision=revision).to_dict()
        layer_plan = P.apply_steps(P.validate(spec.get("plan", {}), config), spec.get("steps", []), config)
    dtype_bytes = {"float16": 2, "bfloat16": 2, "float32": 4}[dtype]
    dev = E.load_target(target) if target else None
    est = E.estimate(config, layer_plan, dev, dtype_bytes=dtype_bytes, context=context, prompt=prompt)
    dense = E.estimate(config, {}, dev, dtype_bytes=dtype_bytes, context=context, prompt=prompt)
    record |= {"model": model, "revision": revision, "dtype": dtype, "plan": P.to_json(layer_plan),
               "estimate": est.to_dict(), "dense": dense.to_dict()}
    typer.echo(f"{model} in {dtype}: {path}")
    typer.echo(f"  weight bytes      {est.weight_bytes:,} ({est.weight_bytes / dense.weight_bytes:.4f} of dense)")
    typer.echo(f"  read per token    {est.decode_bytes_per_token:,} at a {context}-token context")
    typer.echo(f"  prompt FLOPs      {est.prefill_flops:.4g} for {prompt} tokens")
    weights_file = path.parent / loader.WEIGHTS
    if is_saved and weights_file.exists():
        size = weights_file.stat().st_size
        diff = (est.weight_bytes - size) / size
        record["file"] = {"path": str(weights_file), "bytes": size, "relative_difference": diff}
        typer.echo(f"  saved file        {size:,} (estimate {diff:+.4%})")
    if dev:
        c = E.check(dev, config, layer_plan, dtype_bytes=dtype_bytes, context=context, prompt=prompt)
        record["check"] = c.to_dict()
        typer.echo(f"  on {dev.name}: decode {est.est_decode_tok_s:.1f} tokens/s (unpruned "
                   f"{dense.est_decode_tok_s:.1f}), prompt {est.est_prefill_tok_s:.0f} tokens/s; "
                   f"fits {c.fits_memory} (over by {c.memory_gap_gb:.3f} GB), speed met {c.meets_speed} "
                   f"(short by {c.speed_gap_tok_s:.1f} tokens/s)")
    results.mkdir(parents=True, exist_ok=True)
    name = path.parent.name if is_saved else path.stem
    suffix = f"__{target}" if target in E.PRESETS else ""
    out = results / f"{d.slug(model)}__{name}__{dtype}{suffix}.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    typer.echo(f"Saved {out}")


@app.command("speed")
def speed_cmd(
    path: Annotated[Path | None, typer.Option(help="Saved pruned model directory (model.safetensors + plan.json).")] = None,
    model: Annotated[str, typer.Option(help="Hugging Face model id (used when --path is not given).")] = (
        "Qwen/Qwen3-0.6B-Base"),
    dtype: Annotated[str, typer.Option(help="float16, bfloat16 or float32 (unpruned model only).")] = "float16",
    prompt_len: Annotated[int, typer.Option(help="Prompt tokens (WikiText-2 validation).")] = 512,
    gen: Annotated[int, typer.Option(help="New tokens generated, exactly (no early stop).")] = 128,
    repeats: Annotated[int, typer.Option(help="Timed runs.")] = 5,
    warmup: Annotated[int, typer.Option(help="Untimed runs first.")] = 3,
    tag: Annotated[str | None, typer.Option(help="Suffix for the results file (e.g. to keep two invocations).")] = None,
    device: Annotated[str | None, typer.Option(help="cuda, mps or cpu (default: best available).")] = None,
    data_dir: Annotated[Path | None, typer.Option(help="Data cache (default: $PRUNEAGENT_CACHE or ./data_cache).")] = None,
    results: Annotated[Path, typer.Option(help="Directory for the results JSON.")] = Path("results/speed"),
) -> None:
    """Measured prefill and decode tokens/s and peak memory (batch 1, fixed prompt, greedy, fixed length)."""
    import json

    from pruneagent import data as d
    from pruneagent import evaluate as ev
    from pruneagent import loader, speed

    device = device or ev.pick_device()
    if path is not None:
        net = loader.load(path, device=device)
        model_id, revision, label = net.pruneagent["model_id"], net.pruneagent["revision"], path.name
        dtype = str(next(net.parameters()).dtype).removeprefix("torch.")
        file_bytes = (path / loader.WEIGHTS).stat().st_size
    else:
        manifest = d.read_manifest(data_dir)
        model_id, label, file_bytes = model, "dense", None
        revision = manifest["entries"][d.wikitext_name("validation", model)]["tokenizer"]["revision"]
        net = ev.load_model(model, revision, dtype, device)
    typer.echo(f"{label}: {model_id}@{(revision or '')[:8]} on {device} in {dtype}; "
               f"prompt {prompt_len}, gen {gen}, warmup {warmup}, repeats {repeats}")
    prompt = d.load_tokens(d.wikitext_name("validation", model_id), data_dir)
    r = speed.measure(net, device, prompt, prompt_len=prompt_len, gen=gen, repeats=repeats, warmup=warmup)
    record = {"label": label, **_run_record(model_id, revision, device, dtype, prompt_len),
              "path": str(path) if path else None, "file_bytes": file_bytes,
              "prompt": f"first {prompt_len} tokens of {d.wikitext_name('validation', model_id)}",
              "device_name": speed.machine_info(device)["device_name"], **r}
    record.pop("seqlen")
    for key in ("prefill_tok_s", "decode_tok_s"):
        s = r[key]
        typer.echo(f"  {key:<14} median {s['median']:9.1f}  min {s['min']:9.1f}  max {s['max']:9.1f}  "
                   f"spread {s['spread']:.2%}")
    if r["peak_memory_bytes"] is not None:
        typer.echo(f"  peak memory    {r['peak_memory_bytes'] / 1e9:.3f} GB  "
                   f"(weights {r['weight_bytes'] / 1e9:.3f} GB)")
    results.mkdir(parents=True, exist_ok=True)
    suffix = f"__{tag}" if tag else ""
    out = results / f"{d.slug(model_id)}__{label}__{device}__{dtype}{suffix}.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    typer.echo(f"Saved {out}")


@app.command("probe")
def probe_cmd(
    model: Annotated[str, typer.Option(help="Hugging Face model id.")] = "Qwen/Qwen3-0.6B-Base",
    dtype: Annotated[str, typer.Option(help="float16, bfloat16 or float32.")] = "float16",
    samples: Annotated[int, typer.Option(help="C4 calibration samples for the probes.")] = 8,
    steps: Annotated[int, typer.Option(help="Probe 1..steps steps of each strategy.")] = 1,
    ground_truth: Annotated[bool, typer.Option(help="Also measure validation perplexity after one step of "
                                               "each strategy on each layer, and compare rankings.")] = False,
    seqlen: Annotated[int, typer.Option(help="Evaluation window length.")] = 2048,
    windows: Annotated[int | None, typer.Option(help="Validation windows to evaluate (default: all).")] = None,
    layers: Annotated[str | None, typer.Option(help="Comma-separated layers (default: all).")] = None,
    scores_path: Annotated[Path | None, typer.Option("--scores", help="Cached scores (computed if missing).")] = None,
    device: Annotated[str | None, typer.Option(help="cuda, mps or cpu (default: best available).")] = None,
    data_dir: Annotated[Path | None, typer.Option(help="Data cache (default: $PRUNEAGENT_CACHE or ./data_cache).")] = None,
    out: Annotated[Path | None, typer.Option(help="Results JSON (default: results/probes/<model>__<device>"
                                             "__<dtype>.json). Rewritten after every evaluation; "
                                             "evaluations already in it are reused.")] = None,
) -> None:
    """Layer probes for every strategy (the sensitivity table); with --ground-truth, check them against
    full-model validation perplexity (assumption A1)."""
    import json
    import time

    from pruneagent import data as d
    from pruneagent import evaluate as ev
    from pruneagent import plan as P
    from pruneagent import probe as pr
    from pruneagent import pruner

    device = device or ev.pick_device()
    revision = d.read_manifest(data_dir)["entries"][d.calib_name(model)]["tokenizer"]["revision"]
    out = out or Path("results/probes") / f"{d.slug(model)}__{device}__{dtype}.json"
    typer.echo(f"{model}@{revision[:8]} on {device} in {dtype}")
    net = ev.load_model(model, revision, dtype, device)
    scores = _scores(net, model, revision, device, dtype, data_dir, scores_path)
    layer_ids = [int(x) for x in layers.split(",")] if layers else list(range(net.config.num_hidden_layers))

    ev.synchronize(device)
    start = time.perf_counter()
    acts = pr.capture(net, d.load_tokens(d.calib_name(model), data_dir)[:samples], device)
    ev.synchronize(device)
    capture_s = time.perf_counter() - start
    start = time.perf_counter()
    table = pr.sensitivity_table(net, acts, scores, steps=tuple(range(1, steps + 1)), layers=layer_ids)
    ev.synchronize(device)
    probe_s = time.perf_counter() - start
    del acts
    typer.echo(f"  probes: capture {capture_s:.1f} s, {len(layer_ids)} layers x {len(P.STRATEGIES)} "
               f"strategies x {steps} steps in {probe_s:.1f} s")

    record = {"method": "probe", **_run_record(model, revision, device, dtype, seqlen),
              "probe_samples": samples, "probe_calib": d.calib_name(model), "scores": scores.meta,
              "capture_seconds": round(capture_s, 2), "probe_seconds": round(probe_s, 2),
              "table": {str(i): lp for i, lp in table.items()}}
    previous = json.loads(out.read_text()) if out.exists() else {}

    def save() -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2) + "\n")

    save()
    if ground_truth:
        gt = previous.get("ground_truth", {}) if previous.get("windows") == windows else {}
        record["windows"] = windows
        record["ground_truth"] = gt
        if "dense" not in gt:
            gt["dense"] = _val_ppl(net, model, seqlen, windows, device, data_dir)
            save()
        typer.echo(f"  dense validation ppl {gt['dense']['ppl']:.4f} ({gt['dense']['n_windows']} windows)")
        dense = P.dense_plan(net.config)
        for i in layer_ids:
            for s in P.STRATEGIES:
                key = f"{i}/{s}"
                if key in gt:
                    continue
                pruned = pruner.prune(net, P.step(dense, i, s, net.config), scores)
                gt[key] = _val_ppl(pruned, model, seqlen, windows, device, data_dir)
                del pruned
                save()
                typer.echo(f"  layer {i:2d} {s:24s} ppl {gt[key]['ppl']:.4f}  ({gt[key]['seconds']} s)")
        ppl = {(int(k.split("/")[0]), k.split("/")[1]): v["ppl"] for k, v in gt.items() if k != "dense"}
        record["comparison"] = pr.compare(table, gt["dense"]["ppl"], ppl)
        save()
        c = record["comparison"]
        typer.echo(f"  Spearman, all {c['n']} pairs: {c.get('all', float('nan')):.3f}")
        for s, v in c["per_strategy"].items():
            typer.echo(f"  Spearman, {s:24s} {v['spearman']:.3f}  (Block Influence {v['spearman_block_influence']:.3f})")
    typer.echo(f"Saved {out}")


@app.command("device-test")
def device_test_cmd(
    name: Annotated[str, typer.Option(help="Target name, e.g. m4 or t4 (also the file name).")],
    device: Annotated[str | None, typer.Option(help="cuda, mps or cpu (default: best available).")] = None,
    copy_mb: Annotated[int, typer.Option(help="Size of the copied buffer in MiB.")] = 1024,
    matmul_n: Annotated[int, typer.Option(help="Square fp16 matmul size.")] = 4096,
    repeats: Annotated[int, typer.Option(help="Timed repeats of each test.")] = 5,
    results: Annotated[Path, typer.Option(help="Directory for the device-target JSON.")] = Path(
        "results/speed/devices"),
) -> None:
    """Measure the device's memory bandwidth (copy) and fp16 compute (matmul); save them as a device target."""
    import json

    from pruneagent import evaluate as ev
    from pruneagent import speed

    device = device or ev.pick_device()
    test = speed.device_test(device, copy_mb=copy_mb, matmul_n=matmul_n, repeats=repeats)
    target = speed.device_target(name, device, test)
    typer.echo(f"{name} ({target['device_name']}, {device}): bandwidth {target['bandwidth_gbs']} GB/s "
               f"(spread {test['bandwidth']['spread']:.2%}), compute {target['compute_tflops']} TFLOPs "
               f"(spread {test['compute']['spread']:.2%})")
    results.mkdir(parents=True, exist_ok=True)
    out = results / f"{name}.json"
    out.write_text(json.dumps(target, indent=2) + "\n")
    typer.echo(f"Saved {out}")


@app.callback()
def main() -> None:
    """PruneAgent command line."""


@app.command("speed-check")
def speed_check_cmd(
    device: Annotated[str, typer.Option(help="Device the measurements were taken on (mps, cuda); selects their files.")],
    targets: Annotated[list[str], typer.Option("--target", help="Device target to estimate against: preset (m4, t4) "
                                               "or device JSON (results/speed/devices/m4.json). Repeat for several.")],
    machine: Annotated[str, typer.Option(help="Machine name for the results file, e.g. m4 or t4.")],
    measured: Annotated[Path, typer.Option(help="Directory of speed-meter results.")] = Path(
        "results/speed_check/measured"),
    plans: Annotated[Path, typer.Option(help="Directory holding the plans named by the measurements' labels.")] = Path(
        "plans"),
    results: Annotated[Path, typer.Option(help="Directory for the results JSON.")] = Path("results/speed_check"),
) -> None:
    """Estimated vs measured decode and prefill tokens/s over the measured plans: Spearman and % error (A6, A8)."""
    import json
    from dataclasses import asdict

    from transformers import AutoConfig

    from pruneagent import estimate as E
    from pruneagent import plan as P
    from pruneagent import speed_check as S

    files = sorted(measured.glob(f"*__{device}__*.json"))
    if len(files) < 2:
        raise typer.BadParameter(f"need at least two measurements for {device} in {measured}")
    devs = {t: E.load_target(t) for t in targets}
    configs: dict = {}
    rows = []
    for f in files:
        m = json.loads(f.read_text())
        key = (m["model"], m["revision"])
        if key not in configs:
            configs[key] = AutoConfig.from_pretrained(m["model"], revision=m["revision"]).to_dict()
        config = configs[key]
        spec = json.loads((plans / f"{m['label']}.json").read_text())
        layer_plan = P.apply_steps(P.validate(spec.get("plan", {}), config), spec.get("steps", []), config)
        dtype_bytes = {"float16": 2, "bfloat16": 2, "float32": 4}[m["dtype"]]
        row = {"plan": m["label"], "measured_file": str(f), "device_name": m["device_name"],
               "measured": {"decode": m["decode_tok_s"]["median"], "prefill": m["prefill_tok_s"]["median"]},
               "measured_spread": {"decode": m["decode_tok_s"]["spread"], "prefill": m["prefill_tok_s"]["spread"]},
               "weight_bytes": {"measured": m["weight_bytes"]}, "estimated": {}}
        for t, dev in devs.items():
            e = E.estimate(config, layer_plan, dev, dtype_bytes=dtype_bytes, prompt=m["settings"]["prompt_len"])
            row["estimated"][t] = {"decode": e.est_decode_tok_s, "prefill": e.est_prefill_tok_s}
            row["weight_bytes"]["estimated"] = e.weight_bytes
        rows.append(row)
    summary = S.compare(rows, targets)
    record = {"machine": machine, "device": device, "targets": {t: asdict(dev) for t, dev in devs.items()},
              "plans": rows, "summary": summary,
              "definitions": {"error": "(estimated - measured) / measured; positive = estimate too fast",
                              "fit_overhead": "in-sample least squares, measured s/token = estimated s/token + "
                                              "overhead; an extra column only, not used for A6 / A8",
                              "fit": "in-sample least-squares line, measured s/token = scale x estimated s/token "
                                     "+ overhead; an extra column only, not used for A6 / A8"}}
    for t in targets:
        typer.echo(f"{machine} against {t} ({devs[t].bandwidth_gb_s} GB/s, {devs[t].compute_tflops} TFLOPs):")
        typer.echo(f"  {'plan':<26}{'decode est':>11}{'meas':>8}{'err':>8}{'prefill est':>13}{'meas':>8}{'err':>8}")
        for i, r in enumerate(rows):
            typer.echo(f"  {r['plan']:<26}{r['estimated'][t]['decode']:>11.1f}{r['measured']['decode']:>8.1f}"
                       f"{summary[t]['decode']['error'][i]:>+8.0%}{r['estimated'][t]['prefill']:>13.0f}"
                       f"{r['measured']['prefill']:>8.0f}{summary[t]['prefill']['error'][i]:>+8.0%}")
        for ms in S.MEASURES:
            s, fit, oh = summary[t][ms], summary[t][ms]["fit"], summary[t][ms]["fit_overhead"]
            rho = "n/a" if s["spearman"] is None else f"{s['spearman']:.3f}"
            typer.echo(f"  {ms:<8} Spearman {rho}  mean |error| {s['mean_abs_error']:.1%}  max |error| "
                       f"{s['max_abs_error']:.1%}")
            typer.echo(f"    in-sample corrections: + {oh['overhead_s_per_token'] * 1e3:.3f} ms/token -> mean "
                       f"{oh['mean_abs_error']:.1%}, max {oh['max_abs_error']:.1%};  x {fit['scale']:.2f} + "
                       f"{fit['overhead_s_per_token'] * 1e3:.3f} ms/token -> mean {fit['mean_abs_error']:.1%}, "
                       f"max {fit['max_abs_error']:.1%}")
    results.mkdir(parents=True, exist_ok=True)
    out = results / f"{machine}.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    typer.echo(f"Saved {out}")


@app.command("agent-round")
def agent_round_cmd(
    model: Annotated[str, typer.Option(help="Hugging Face model id.")] = "Qwen/Qwen3-0.6B-Base",
    probes: Annotated[Path, typer.Option(help="Quick-test (probe) table JSON from `pruneagent probe`.")] = Path(
        "results/probes/qwen3-0.6b-base__cuda__float16.json"),
    memory_cut: Annotated[float, typer.Option(help="Memory limit: this share less weight memory than unpruned.")] = 0.10,
    reply: Annotated[Path | None, typer.Option(help="Reuse a saved Codex reply instead of calling Codex.")] = None,
    measure: Annotated[bool, typer.Option(help="Build each valid candidate and measure it on the GPU.")] = True,
    baseline_only: Annotated[bool, typer.Option(
        help="Only add Baseline: FLAP at the same memory limit to an existing output JSON.")] = False,
    scores_path: Annotated[Path | None, typer.Option("--scores", help="Cached scores (computed if missing).")] = None,
    reference: Annotated[Path, typer.Option(help="Subset results with Wanda-50% on the 32 windows.")] = Path(
        "results/subset/qwen3-0.6b-base__cuda__float16.json"),
    dtype: Annotated[str, typer.Option(help="float16, bfloat16 or float32.")] = "float16",
    seqlen: Annotated[int, typer.Option(help="Evaluation window length.")] = 2048,
    device: Annotated[str | None, typer.Option(help="cuda, mps or cpu (default: best available).")] = None,
    data_dir: Annotated[Path | None, typer.Option(help="Data cache (default: $PRUNEAGENT_CACHE or ./data_cache).")] = None,
    out: Annotated[Path, typer.Option(help="Output JSON (a .md summary is written next to it).")] = Path(
        "results/agent_round/round1.json"),
    previous: Annotated[Path | None, typer.Option(
        help="Earlier output JSON of the same round; its perplexities are shown next to the new ones.")] = None,
    previous_calibration: Annotated[str, typer.Option(
        help="How the earlier run's FLAP scores were calibrated (label for --previous).")] = "WikiText-2 train",
    title: Annotated[str | None, typer.Option(help="Heading of the .md summary.")] = None,
) -> None:
    """One agent round by hand: prompt, one Codex call, parse, validate, apply one step per choice, measure.
    Call Codex once with --no-measure, then measure from the saved reply under the GPU lock:
    scripts/mac_gpu.sh uv run pruneagent agent-round --reply results/agent_round/round1_reply.txt"""
    import gc
    import json
    import time

    from transformers import AutoConfig

    from pruneagent import agent_round as A
    from pruneagent import data as d
    from pruneagent import estimate as E
    from pruneagent import plan as P

    manifest = d.read_manifest(data_dir)
    revision = manifest["entries"][d.calib_name(model)]["tokenizer"]["revision"]
    config = AutoConfig.from_pretrained(model, revision=revision)
    table = json.loads(probes.read_text())["table"]
    dense_bytes = E.weight_bytes(config, P.dense_plan(config))
    limit = A.memory_limit(config, memory_cut)

    if baseline_only:
        from pruneagent import evaluate as ev

        record = json.loads(out.read_text())
        device = device or ev.pick_device()
        net = ev.load_model(model, revision, dtype, device)
        scores = _scores(net, model, revision, device, dtype, data_dir, scores_path)
        ids = d.load_tokens(d.wikitext_name("validation", model), data_dir).reshape(-1)
        limit = record["memory"]["limit_bytes"]
        record["baseline_flap"] = A.measure_flap(net, scores, config, limit, ids, seqlen, device, ev.GATE_WINDOWS)
        record["baseline_flap"] |= {"device": device, "dtype": dtype, "scores": scores.meta}
        b = record["baseline_flap"]
        typer.echo(f"  Baseline: FLAP ratio {b['ratio']}: ppl {b['ppl']:.4f}, removed "
                   f"{b['weight_bytes_removed'] / 1e6:,.1f} MB, within limit {b['within_limit']}")
        out.write_text(json.dumps(record, indent=2) + "\n")
        out.with_suffix(".md").write_text(A.summary_markdown(record))
        typer.echo(f"Saved {out}")
        return

    prompt = A.build_prompt(config, table, limit)
    out.parent.mkdir(parents=True, exist_ok=True)
    names = ("prompt.txt", "reply.txt", "codex.json", "codex_stderr.txt")
    side = {k: out.with_name(f"{out.stem}_{k}") for k in names}
    if reply is not None:  # read the saved call's files next to the reply (round1_reply.txt -> round1_prompt.txt)
        stem = reply.name.removesuffix("reply.txt")
        side = {k: reply.with_name(stem + k) for k in names} if stem != reply.name else side
    if reply is not None and side["prompt.txt"].exists():
        prompt = side["prompt.txt"].read_text().removesuffix("\n")  # the prompt that produced the saved reply
    else:
        side["prompt.txt"].write_text(prompt + "\n")
    typer.echo(f"Prompt: {len(prompt):,} characters, {len(prompt.split()):,} words")

    if reply is None:
        typer.echo("Calling Codex (one call; usually under a minute)...")
        codex = A.call_codex(prompt)
        side["reply.txt"].write_text(codex["stdout"])
        side["codex_stderr.txt"].write_text(codex["stderr"])
        side["codex.json"].write_text(
            json.dumps({k: v for k, v in codex.items() if k not in ("stdout", "stderr")}, indent=2) + "\n")
        typer.echo(f"  exit {codex['returncode']} in {codex['seconds']} s; tokens used {codex['tokens_used']}; "
                   f"reply saved to {side['reply.txt']}")
        if codex["returncode"] != 0:
            raise typer.Exit(1)
    else:
        codex = json.loads(side["codex.json"].read_text()) if side["codex.json"].exists() else {}
        codex["stderr"] = side["codex_stderr.txt"].read_text() if side["codex_stderr.txt"].exists() else None
        codex["stdout"] = reply.read_text()
    raw = codex["stdout"]

    record = {"task": "1.11", "model": model, "revision": revision, "probes": str(probes), "prompt": prompt,
              "prompt_chars": len(prompt), "prompt_words": len(prompt.split()), "codex": codex, "raw_reply": raw,
              "memory": {"dense_bytes": dense_bytes, "limit_bytes": limit, "cut": memory_cut,
                         "step_savings_bytes": A.step_savings(config)},
              "calibration": {"entry": d.calib_name(model), "sha256": manifest["entries"][d.calib_name(model)]["sha256"],
                              "used_by": "FLAP scores (Baseline: FLAP, narrow_mlp, remove_attention_group) and Wanda"}}
    if reply is not None:
        record["reply_source"] = str(reply)
    if title:
        record["title"] = title
    if previous is not None:
        old = json.loads(previous.read_text())
        record["previous"] = A.previous_numbers(old, str(previous), previous_calibration)
        if old.get("prompt") == prompt and old.get("prompt_tokens_est"):
            record |= {k: old[k] for k in ("prompt_tokens_est", "prompt_tokens_note") if k in old}
    try:
        cands = A.parse_reply(raw)
    except (ValueError, TypeError) as e:
        record |= {"parse_error": str(e), "candidates": [], "validation": []}
        out.write_text(json.dumps(record, indent=2) + "\n")
        typer.echo(f"Could not parse the reply: {e}. Saved {out}")
        raise typer.Exit(1) from e
    checks = A.validate_candidates(cands, config, limit_bytes=limit)
    record |= {"candidates": cands, "validation": checks}
    for c, v in zip(cands, checks, strict=True):
        typer.echo(f"  candidate {v['index'] + 1}: {len(c['choices'] or [])} choices, "
                   + (f"valid, weights {v['weight_bytes'] / 1e6:,.1f} MB (limit {limit / 1e6:,.1f})" if v["valid"]
                      else "INVALID: " + "; ".join(v["reasons"])))

    if reference.exists():
        record["reference_ppl_32"] = json.loads(reference.read_text())["plans"]["p00_wanda50_all"]["ppl"]["32"]
        record["reference_source"] = f"{reference} (Wanda-50% on every layer, 32-window subset)"

    if measure and any(v["valid"] for v in checks):
        from pruneagent import evaluate as ev
        from pruneagent import pruner

        device = device or ev.pick_device()
        record |= _run_record(model, revision, device, dtype, seqlen)
        net = ev.load_model(model, revision, dtype, device)
        scores = _scores(net, model, revision, device, dtype, data_dir, scores_path)
        record["scores"] = scores.meta
        ids = d.load_tokens(d.wikitext_name("validation", model), data_dir).reshape(-1)
        r = ev.subset_perplexity(net, ids, ev.GATE_WINDOWS, seqlen, device)
        record["dense_ppl_32"] = r.ppl
        typer.echo(f"  unpruned: ppl {r.ppl:.4f} on {r.n_windows} windows ({r.seconds:.0f} s)")
        record["baseline_flap"] = A.measure_flap(net, scores, config, limit, ids, seqlen, device, ev.GATE_WINDOWS)
        typer.echo(f"  Baseline: FLAP: ppl {record['baseline_flap']['ppl']:.4f}")
        for v in checks:
            if not v["valid"]:
                continue
            start = time.perf_counter()
            pruned = pruner.prune(net, v["plan"], scores)
            r = ev.subset_perplexity(pruned, ids, ev.GATE_WINDOWS, seqlen, device)
            v["measured"] = {"ppl": r.ppl, "n_windows": r.n_windows, "eval_seconds": round(r.seconds, 1),
                             "total_seconds": round(time.perf_counter() - start, 1),
                             "parameters": pruner.weight_counts(pruned)["parameters"],
                             "weight_bytes_removed": dense_bytes - v["weight_bytes"],
                             "within_limit": v["weight_bytes"] <= limit}
            typer.echo(f"  candidate {v['index'] + 1}: ppl {r.ppl:.4f} ({r.seconds:.0f} s)")
            del pruned
            gc.collect()
    out.write_text(json.dumps(record, indent=2) + "\n")
    if "reference_ppl_32" in record:
        out.with_suffix(".md").write_text(A.summary_markdown(record))
    typer.echo(f"Saved {out}")


@app.command("runs")
def runs_cmd(
    root: Annotated[Path, typer.Option(help="Directory holding the runs.")] = Path("runs"),
) -> None:
    """List runs with their status (running, interrupted, paused and why, finished), last round and accepted
    plans."""
    from pruneagent import runstore

    found = runstore.list_runs(root)
    if not found:
        typer.echo(f"No runs under {root}.")
        return
    width = max(len(r.run_id) for r in found)
    typer.echo(f"{'run':<{width}}  {'round':>5}  {'accepted':>8}  {'updated (UTC)':<19}  status")
    for r in found:
        typer.echo(f"{r.run_id:<{width}}  {r.round:>5}  {r.n_accepted:>8}  {r.updated[:19]:<19}  {r.describe()}")


@app.command("resume")
def resume_cmd(
    run_id: Annotated[str | None, typer.Argument(help="Run to resume; optional when exactly one run is paused.")]
    = None,
    root: Annotated[Path, typer.Option(help="Directory holding the runs.")] = Path("runs"),
    backend: Annotated[str | None, typer.Option(
        help="Continue on this agent backend (codex or api) instead of the run's own.")] = None,
) -> None:
    """Continue a paused or interrupted run from its last checkpoint, without repeating an agent call. A run
    paused at a usage limit that has not reset yet pauses again at once, without a call."""
    from pruneagent import graph, runstore

    if backend not in (None, "codex", "api"):
        typer.echo("--backend must be codex or api", err=True)
        raise typer.Exit(1)
    try:
        state = graph.resume_run(run_id, root, backend)
    except runstore.ResumeError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None
    if state.get("paused"):
        typer.echo(f"Paused after round {state.get('round', 0)}: {state['paused']}")
        return
    typer.echo(f"Finished after round {state.get('round', 0)}: {state.get('stop_reason')}")


@app.command("limits")
def limits_cmd(
    calls: Annotated[int, typer.Option(help="Agent calls the run would make.")] = 60,
    ledger: Annotated[Path, typer.Option(help="The agent usage ledger.")] = Path("runs/agent_usage.jsonl"),
) -> None:
    """Show the latest Codex usage-limit reading and whether a run of CALLS agent calls fits (no call is made)."""
    from pruneagent import agent_client as AC

    client = AC.AgentClient(AC.CodexBackend(), ledger=AC.Ledger(ledger))
    reading = client.limit_reading()
    if reading is not None:
        typer.echo(f"Latest reading ({reading.time.astimezone():%Y-%m-%d %H:%M}): {reading.summary()}")
    typer.echo(client.estimate_run(calls).message())
