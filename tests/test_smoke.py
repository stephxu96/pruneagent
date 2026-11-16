"""Smoke tests: the package and its key dependencies import, and the CLI runs."""

import importlib

import pytest
from typer.testing import CliRunner

import pruneagent
from pruneagent.cli import app


def test_version():
    assert pruneagent.__version__ == "0.1.0"


def test_cli_version():
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == pruneagent.__version__


@pytest.mark.parametrize("module", ["torch", "transformers", "datasets", "safetensors", "langgraph", "lm_eval"])
def test_dependency_imports(module):
    importlib.import_module(module)
