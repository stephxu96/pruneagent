"""Command-line entry point. Subcommands (bar, run, resume, baselines, report) arrive in later tasks."""

import typer

from pruneagent import __version__

app = typer.Typer(help="PruneAgent: prune Qwen3 to Wanda-level quality with real savings.", no_args_is_help=True)


@app.command()
def version() -> None:
    """Print the PruneAgent version."""
    typer.echo(__version__)


@app.callback()
def main() -> None:
    """PruneAgent command line."""
