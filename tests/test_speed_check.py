"""Estimate-vs-measured speed check: rank correlation, errors, the in-sample fit, and the speed-check command."""

import json
from pathlib import Path

import pytest
from transformers import AutoConfig, Qwen3Config
from typer.testing import CliRunner

from pruneagent import estimate as E
from pruneagent import speed
from pruneagent import speed_check as S
from pruneagent.cli import app

ROOT = Path(__file__).resolve().parents[1]


def test_ranks_average_ties():
    assert S.ranks([10, 30, 20, 20]) == [1, 4, 2.5, 2.5]


def test_spearman():
    assert S.spearman([1, 2, 3, 4], [10, 20, 30, 1000]) == pytest.approx(1)
    assert S.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1)
    assert S.spearman([1, 2, 3], [5, 5, 5]) is None
    # one swapped pair among four: 1 - 6 * 2 / (4 * 15)
    assert S.spearman([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)


def test_score_errors():
    s = S.score([110, 50], [100, 100])
    assert s["error"] == pytest.approx([0.1, -0.5])
    assert s["mean_abs_error"] == pytest.approx(0.3)
    assert s["max_abs_error"] == pytest.approx(0.5)


def test_fit_recovers_scale_and_overhead():
    est = [50.0, 80.0, 120.0, 200.0]
    meas = [1 / (1.5 / v + 0.004) for v in est]  # 1.5x slower per token plus 4 ms
    fit = S.fit_time(est, meas)
    assert fit["scale"] == pytest.approx(1.5)
    assert fit["overhead_s_per_token"] == pytest.approx(0.004)
    assert fit["corrected"] == pytest.approx(meas)
    assert fit["max_abs_error"] == pytest.approx(0, abs=1e-9)


def test_fit_overhead_recovers_fixed_overhead():
    est = [50.0, 80.0, 120.0]
    meas = [1 / (1 / v + 0.003) for v in est]
    fit = S.fit_overhead(est, meas)
    assert fit["overhead_s_per_token"] == pytest.approx(0.003)
    assert fit["max_abs_error"] == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize("total", [None, 25.8])
def test_load_target_reads_device_test_file(tmp_path, monkeypatch, total):
    # machine_info depends on the OS (sysctl is missing on Linux CI), so fix what it reports
    monkeypatch.setattr(speed, "machine_info", lambda device: {"device_name": "test", "memory_total_gb": total})
    test = {"bandwidth": {"median": 94.8}, "compute": {"median": 3.48}}
    for name, limit in [("m4", E.PRESETS["m4"].memory_limit_gb), ("other", total)]:
        f = tmp_path / f"{name}.json"
        f.write_text(json.dumps(speed.device_target(name, "mps", test)))
        if limit is None:
            with pytest.raises(ValueError, match="no memory_limit_gb"):
                E.load_target(str(f))
            continue
        t = E.load_target(str(f))
        assert (t.bandwidth_gb_s, t.compute_tflops, t.backend, t.memory_limit_gb) == (94.8, 3.48, "gpu", limit)


def test_load_target_reads_committed_device_files():
    for name in ("m4", "t4"):
        t = E.load_target(str(ROOT / "results/speed/devices" / f"{name}.json"))
        assert t.memory_limit_gb == E.PRESETS[name].memory_limit_gb
        assert t.bandwidth_gb_s < E.PRESETS[name].bandwidth_gb_s


def _measurement(label, decode, prefill):
    return {"label": label, "model": "tiny", "revision": "r", "dtype": "float16", "device_name": "test",
            "settings": {"prompt_len": 32, "gen": 8}, "weight_bytes": 1,
            "decode_tok_s": {"median": decode, "spread": 0.01}, "prefill_tok_s": {"median": prefill, "spread": 0.01}}


def test_speed_check_command(tmp_path, monkeypatch):
    cfg = Qwen3Config(vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=4,
                      num_attention_heads=8, num_key_value_heads=4, head_dim=16)
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *a, **k: cfg)
    plans, measured = tmp_path / "plans", tmp_path / "measured"
    plans.mkdir()
    measured.mkdir()
    for name, plan, dec in [("a", {}, 10.0), ("b", {"0": {"dropped": True}}, 12.0),
                            ("c", {"0": {"dropped": True}, "1": {"dropped": True}}, 15.0)]:
        (plans / f"{name}.json").write_text(json.dumps({"plan": plan}))
        (measured / f"tiny__{name}__cpu__float16.json").write_text(json.dumps(_measurement(name, dec, 100 * dec)))
    r = CliRunner().invoke(app, ["speed-check", "--device", "cpu", "--target", "m4", "--machine", "x",
                                 "--measured", str(measured), "--plans", str(plans), "--results", str(tmp_path)])
    assert r.exit_code == 0, r.output
    out = json.loads((tmp_path / "x.json").read_text())
    assert [p["plan"] for p in out["plans"]] == ["a", "b", "c"]
    assert out["summary"]["m4"]["decode"]["spearman"] == pytest.approx(1)
    est = out["plans"][0]["estimated"]["m4"]["decode"]
    assert out["summary"]["m4"]["decode"]["error"][0] == pytest.approx((est - 10) / 10)


def test_layer_cost_fit_recovers_a_known_cost():
    est = [100.0, 150.0, 200.0]
    kept = [28, 20, 14]
    meas = [1 / (1 / e + 2e-4 * k) for e, k in zip(est, kept, strict=True)]
    fit = S.fit_layer_cost(est, meas, kept)
    assert fit["layer_cost_s"] == pytest.approx(2e-4) and fit["max_abs_error"] < 1e-9


def test_fitted_layer_costs_match_the_saved_measurements():
    """estimate.LAYER_COST_S is the fit on the task 1.7 results in the repository."""
    root = ROOT / "results" / "speed_check"
    fits = {}
    for machine in ("m4", "t4"):
        fits |= S.layer_costs(root / f"{machine}.json", root / "prune_log")
    assert set(fits) == set(E.LAYER_COST_S)
    for name, f in fits.items():
        assert E.LAYER_COST_S[name] == pytest.approx(f["layer_cost_s"], rel=1e-3)
        assert f["max_abs_error"] < 0.10 and f["spearman"] >= 0.9
