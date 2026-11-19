"""Task 3.5: a run becomes report.md + charts + report.json, with numbers that match the run's files."""

import json

import pytest
from test_kill_resume import setup_for  # noqa: F401  (fixture)
from test_web import finished_run  # noqa: F401  (fixture)

pytest.importorskip("matplotlib")

from pruneagent import report


def test_report_builds_from_a_run(finished_run):  # noqa: F811
    md = report.write(finished_run, "r1")
    out = md.parent
    assert {p.name for p in out.iterdir()} >= {"report.md", "report.json", "path.png", "layers.png", "rounds.png"}
    s = json.loads((out / "report.json").read_text())
    text = md.read_text()
    assert s["heldout"]["agent"]["test_error"] == 12.0  # from heldout.json
    assert abs(s["heldout"]["agent"]["quality_loss"] - 0.2) < 1e-9  # 12 / 10 - 1
    assert "PruneAgent is 20% worse than the original, the standard method (FLAP) 50% worse" in text
    assert s["final"]["size_mb"] <= s["limit_mb"]
    kept = sum(v["kept"] for v in s["strategies"].values())
    assert kept >= s["outcomes"].get("ACCEPT", 0)  # every kept round keeps at least one change
    assert text.count("\n| ") > 10 and "## Every round" in text
