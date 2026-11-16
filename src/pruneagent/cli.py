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
    model: Annotated[str, typer.Option(help="Model whose tokenizer is used.")] = "Qwen/Qwen3-0.6B",
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


@app.callback()
def main() -> None:
    """PruneAgent command line."""
