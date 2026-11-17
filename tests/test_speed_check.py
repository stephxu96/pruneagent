"""Estimate-vs-measured speed check: rank correlation, errors, the in-sample fit, and the speed-check command."""

import json

import pytest
from transformers import AutoConfig, Qwen3Config
from typer.testing import CliRunner

from pruneagent import estimate as E
from pruneagent import speed
from pruneagent import speed_check as S
from pruneagent.cli import app


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


def test_load_target_reads_device_test_file(tmp_path):
    test = {"bandwidth": {"median": 94.8}, "compute": {"median": 3.48}}
    f = tmp_path / "m4.json"
    f.write_text(json.dumps(speed.device_target("m4", "mps", test)))
    t = E.load_target(str(f))
    assert (t.bandwidth_gb_s, t.compute_tflops, t.backend) == (94.8, 3.48, "gpu")
    assert t.memory_limit_gb > 0


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
