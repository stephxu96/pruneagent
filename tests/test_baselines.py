import json

import pytest
import torch

from pruneagent import baselines as B
from pruneagent import flap


class Counter:
    """Fake evaluator: value = f(ratio), counting calls."""

    def __init__(self, f):
        self.f, self.calls = f, []

    def __call__(self, r):
        self.calls.append(r)
        return self.f(r)


def ppl_curve(r):
    return 15 + 40 * r  # crosses 26.05 at r = 0.27625


def test_bisect_finds_largest_passing_ratio_within_tolerance():
    ev = Counter(ppl_curve)
    ratio, trials = B.bisect_ratio(ev, 26.05, lo=0.0, hi=0.5, tol=0.005)
    assert ppl_curve(ratio) <= 26.05
    assert 0.27625 - ratio <= 0.005
    assert ev.calls[0] == 0.5  # hi checked first
    assert len(ev.calls) <= 9
    # every passing trial is at or below the answer, every failing one above it
    assert all(t.ratio <= ratio for t in trials if t.passed)
    assert all(t.ratio > ratio for t in trials if not t.passed)


def test_bisect_returns_hi_when_it_passes():
    ev = Counter(ppl_curve)
    ratio, trials = B.bisect_ratio(ev, 100.0, hi=0.5)
    assert ratio == 0.5 and len(trials) == 1 and trials[0].passed


def test_bisect_returns_lo_when_nothing_passes():
    ratio, trials = B.bisect_ratio(lambda r: 99.0, 26.05, lo=0.0, hi=0.5, tol=0.01)
    assert ratio == 0.0 and not any(t.passed for t in trials)


def test_bisect_rejects_bad_interval():
    with pytest.raises(ValueError):
        B.bisect_ratio(ppl_curve, 26.05, lo=0.5, hi=0.2)


def test_search_accepts_when_accuracy_holds():
    acc = Counter(lambda r: 0.50 - 0.2 * r)
    s = B.search_bar(ppl_curve, acc, 26.05, 0.418)
    assert s.meets_bar and s.ratio == acc.calls[0]
    assert len(acc.calls) == 1


def test_search_bisects_accuracy_when_it_fails_at_the_perplexity_ratio():
    acc = Counter(lambda r: 0.52 - 0.4 * r)  # crosses 0.418 at r = 0.255
    s = B.search_bar(ppl_curve, acc, 26.05, 0.418, acc_tol=0.01)
    assert s.meets_bar and 0.255 - 0.01 <= s.ratio <= 0.255
    assert not s.acc_trials[0].passed  # failed at the perplexity ratio (~0.273)
    assert s.ratio == max(t.ratio for t in s.acc_trials if t.passed)
    assert all(t.ratio > s.ratio for t in s.acc_trials if not t.passed)


def test_search_skips_ratio_known_to_fail_accuracy():
    acc = Counter(lambda r: 0.52 - 0.4 * r)
    s = B.search_bar(ppl_curve, acc, 26.05, 0.418, acc_lo=0.1, acc_tol=0.01, acc_fails_at=0.24)
    assert all(r < 0.24 for r in acc.calls)
    assert s.meets_bar and 0.23 <= s.ratio <= 0.24


def test_search_verifies_acc_lo_when_it_is_the_answer():
    acc = Counter(lambda r: 0.30)
    s = B.search_bar(ppl_curve, acc, 26.05, 0.418, acc_lo=0.1, acc_tol=0.05)
    assert not s.meets_bar and s.ratio is None and 0.1 in acc.calls and "accuracy" in s.note


def test_search_gives_up_when_accuracy_never_holds():
    s = B.search_bar(ppl_curve, lambda r: 0.30, 26.05, 0.418, acc_tol=0.02)
    assert not s.meets_bar and s.ratio is None and "accuracy" in s.note


def test_search_from_a_known_interval_checks_lo():
    # resuming from an earlier run's final interval: hi fails, lo is assumed, so it is evaluated once
    ev, acc = Counter(ppl_curve), Counter(lambda r: 0.45)
    s = B.search_bar(ev, acc, 26.05, 0.418, lo=0.2734, hi=0.2773, tol=0.005)
    assert s.meets_bar and s.ratio == 0.2734
    assert ev.calls == [0.2773, 0.2734] and acc.calls == [0.2734]


def test_search_reports_when_assumed_lo_fails():
    s = B.search_bar(lambda r: 99.0, lambda r: 1.0, 26.05, 0.418, lo=0.2, hi=0.21, tol=0.05)
    assert not s.meets_bar and "perplexity" in s.note and len(s.ppl_trials) == 2


def test_search_reports_when_perplexity_never_passes():
    acc = Counter(lambda r: 1.0)
    s = B.search_bar(lambda r: 99.0, acc, 26.05, 0.418, tol=0.05)
    assert not s.meets_bar and not acc.calls and "perplexity" in s.note


def test_bar_from_results(tmp_path):
    (tmp_path / "w.json").write_text(json.dumps({"splits": {"validation": {"ppl": 23.6826}, "test": {"ppl": 22.5}}}))
    (tmp_path / "m.json").write_text(json.dumps({"runs": {"w50": {"mean_acc": 0.4380}}}))
    bar = B.bar_from_results(tmp_path / "w.json", tmp_path / "m.json", "w50")
    assert bar["max_validation_ppl"] == pytest.approx(26.05086)
    assert bar["min_mean_acc"] == pytest.approx(0.418)


def test_flap_from_stats_matches_prune_flap():
    from test_flap import tiny

    calib = torch.randint(0, 97, (8, 16))
    a, b = tiny(), tiny()
    out_a = flap.prune_flap(a, calib, 0.3, "cpu")
    out_b = flap.prune_flap_from_stats(b, flap.collect(tiny(), calib, "cpu"), 0.3)
    assert torch.equal(out_a["attn_keep"], out_b["attn_keep"]) and torch.equal(out_a["mlp_keep"], out_b["mlp_keep"])
    ids = torch.randint(0, 97, (1, 12))
    with torch.no_grad():
        assert torch.allclose(a(ids).logits, b(ids).logits)
    before, after = B.memory(tiny()), B.memory(a)
    assert after["parameters"] < before["parameters"]
    assert after["weight_bytes"] == 4 * after["parameters"]
