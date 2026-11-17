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


@app.callback()
def main() -> None:
    """PruneAgent command line."""
