"""Command-line entry point. Further subcommands (bar, run, resume, baselines, report) arrive in later tasks."""

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
    if scores_path.exists():
        typer.echo(f"  scores from {scores_path}")
        return pruner.load_scores(scores_path)
    start = time.perf_counter()
    manifest = d.read_manifest(data_dir)
    meta = {"model": model, "revision": revision, "device": device, "dtype": dtype,
            "flap_calib": d.flap_calib_name(model),
            "flap_calib_sha256": manifest["entries"][d.flap_calib_name(model)]["sha256"],
            "wanda_calib": d.calib_name(model),
            "wanda_calib_sha256": manifest["entries"][d.calib_name(model)]["sha256"]}
    scores = pruner.compute_scores(net, d.load_tokens(d.flap_calib_name(model), data_dir),
                                   d.load_tokens(d.calib_name(model), data_dir), device, meta=meta)
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


@app.callback()
def main() -> None:
    """PruneAgent command line."""
