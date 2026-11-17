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


@app.callback()
def main() -> None:
    """PruneAgent command line."""
