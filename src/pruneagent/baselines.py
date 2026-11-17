"""Baselines at the quality bar (task 1.9): the largest pruning ratio a method can use and still meet the bar.

The bar (task 1.1) has two parts, both must hold:
- WikiText-2 validation perplexity <= 1.10 x Wanda-50%'s,
- mean multiple-choice accuracy over the seven question sets >= Wanda-50%'s mean - 2 points.

Search: bisect the ratio on validation perplexity alone (cheap: ~30 s per evaluation on a T4), assuming
perplexity rises with the ratio. The interval [lo, hi] always has lo passing (or untested at 0) and hi failing;
it halves until it is narrower than `tol`, and lo is the answer. Accuracy (~12 min) is then checked only at that
ratio; if it fails, the ratio steps down by `step` (perplexity re-checked) until both parts hold or `max_steps`
is used up.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Trial:
    ratio: float
    value: float
    passed: bool


@dataclass
class BarSearch:
    ratio: float | None  # largest ratio found that meets the whole bar; None if none did
    ppl_trials: list[Trial] = field(default_factory=list)
    acc_trials: list[Trial] = field(default_factory=list)
    meets_bar: bool = False
    note: str = ""

    def to_json(self) -> dict:
        return asdict(self)


def bisect_ratio(
    evaluate: Callable[[float], float], max_value: float, lo: float = 0.0, hi: float = 0.5, tol: float = 0.005,
    digits: int = 4,
) -> tuple[float, list[Trial]]:
    """Largest ratio in [lo, hi] (to within `tol`) whose `evaluate(ratio)` <= `max_value`, assuming the value
    rises with the ratio. `lo` is assumed to pass and is not evaluated; `hi` is evaluated first and returned if
    it passes. Returns (ratio, trials); the ratio is `lo` itself if every evaluated ratio failed."""
    if not 0 <= lo < hi:
        raise ValueError(f"need 0 <= lo < hi, got lo={lo}, hi={hi}")
    trials = []
    v = evaluate(hi)
    trials.append(Trial(hi, v, v <= max_value))
    if v <= max_value:
        return hi, trials
    while hi - lo > tol:
        mid = round((lo + hi) / 2, digits)
        if mid in (lo, hi):
            break
        v = evaluate(mid)
        ok = v <= max_value
        trials.append(Trial(mid, v, ok))
        if ok:
            lo = mid
        else:
            hi = mid
    return lo, trials


def search_bar(
    ppl: Callable[[float], float], acc: Callable[[float], float], max_ppl: float, min_acc: float,
    lo: float = 0.0, hi: float = 0.5, tol: float = 0.005, step: float = 0.02, max_steps: int = 2,
) -> BarSearch:
    """Bisect on perplexity, then check accuracy at the ratio found, stepping down until both hold."""
    ratio, ppl_trials = bisect_ratio(ppl, max_ppl, lo, hi, tol)
    out = BarSearch(ratio=None, ppl_trials=ppl_trials)
    if not any(t.passed for t in ppl_trials):
        out.note = f"no evaluated ratio meets the perplexity bar (bisection stopped at {ratio})"
        return out
    for i in range(max_steps + 1):
        if i:
            ratio = round(ratio - step, 4)
            if ratio <= 0:
                out.note = "accuracy bar not met at any positive ratio tried"
                return out
            v = ppl(ratio)
            out.ppl_trials.append(Trial(ratio, v, v <= max_ppl))
            if v > max_ppl:
                continue
        a = acc(ratio)
        out.acc_trials.append(Trial(ratio, a, a >= min_acc))
        if a >= min_acc:
            out.ratio, out.meets_bar = ratio, True
            return out
    out.note = f"accuracy bar not met after stepping down {max_steps} x {step}"
    return out


def bar_from_results(wanda_ppl_json: Path, mc_json: Path, mc_run: str, ppl_margin: float = 1.10,
                     acc_margin: float = 0.02) -> dict:
    """The quality bar from task 1.1's recorded Wanda-50% results."""
    w = json.loads(Path(wanda_ppl_json).read_text())
    m = json.loads(Path(mc_json).read_text())["runs"][mc_run]
    return {
        "wanda50_validation_ppl": w["splits"]["validation"]["ppl"], "wanda50_test_ppl": w["splits"]["test"]["ppl"],
        "wanda50_mean_acc": m["mean_acc"],
        "max_validation_ppl": ppl_margin * w["splits"]["validation"]["ppl"],
        "min_mean_acc": m["mean_acc"] - acc_margin,
        "sources": {"ppl": str(wanda_ppl_json), "mc": f"{mc_json}#{mc_run}"},
    }


def memory(model) -> dict[str, int]:
    """Parameters and their bytes (tied embeddings counted once; compensation biases included)."""
    params = list(model.parameters())
    return {"parameters": sum(p.numel() for p in params),
            "weight_bytes": sum(p.numel() * p.element_size() for p in params)}
