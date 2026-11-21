"""Compared methods (architecture v0.8).

Primary comparison, Baseline: FLAP: FLAP cut to the same memory limit as the agent. `smallest_fitting_ratio`
finds the smallest pruning ratio whose model fits a weight-memory limit; `flap_at_memory_limit` applies it to FLAP.

Backup, the Wanda-50% match (task 1.9; optional since v0.8): the largest ratio at which FLAP or uniform Wanda-sp
still keeps the quality of the Wanda-50% reference (task 1.1's results). Both parts must hold:
- WikiText-2 validation perplexity <= Wanda-50%'s (times `ppl_margin`, default 1.0: no allowance),
- mean multiple-choice accuracy over the seven question sets >= Wanda-50%'s mean (minus `acc_margin`, default 0).
Task 1.9's committed results used an earlier allowance (1.10 x perplexity, 2 points of accuracy).

Search for the Wanda-50% match: bisect the ratio on validation perplexity alone (cheap: ~30 s per evaluation on a T4), assuming
perplexity rises with the ratio. The interval [lo, hi] always has lo passing (or untested at 0) and hi failing;
it halves until it is narrower than `tol`, and lo is the answer. Accuracy (~12 min) is then checked only at that
ratio; if it fails, a second bisection runs below that ratio on both parts together (perplexity first, accuracy
only where perplexity passes), to within `acc_tol`.
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
class Wanda50MatchSearch:
    ratio: float | None  # largest ratio found that keeps Wanda-50% quality (both parts); None if none did
    ppl_trials: list[Trial] = field(default_factory=list)
    acc_trials: list[Trial] = field(default_factory=list)
    matches_wanda50: bool = False
    note: str = ""

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class AtMemoryLimit:
    limit_bytes: int
    ratio: float | None  # smallest ratio found whose model fits; None if even `hi` does not fit
    weight_bytes: int | None  # at that ratio
    trials: list[Trial] = field(default_factory=list)  # value = weight bytes, passed = fits
    note: str = ""

    def to_json(self) -> dict:
        return asdict(self)


def smallest_fitting_ratio(
    weight_bytes: Callable[[float], int], limit_bytes: int, hi: float = 0.6, tol: float = 0.001, digits: int = 4,
) -> AtMemoryLimit:
    """Smallest pruning ratio in [0, hi] (to within `tol`) whose model's weight bytes are <= `limit_bytes`,
    assuming bytes fall as the ratio rises. Ratio 0 (the unpruned model) is checked first: if it fits, nothing
    needs pruning. Then `hi`; if it does not fit, the limit is out of reach for this method."""
    out = AtMemoryLimit(limit_bytes=limit_bytes, ratio=None, weight_bytes=None)

    def fits(r: float) -> bool:
        b = weight_bytes(r)
        out.trials.append(Trial(r, b, b <= limit_bytes))
        if b <= limit_bytes:
            out.ratio, out.weight_bytes = r, b
        return b <= limit_bytes

    if fits(0.0):
        out.note = "the unpruned model already fits"
        return out
    if not fits(hi):
        out.note = f"does not fit even at ratio {hi}"
        return out
    lo = 0.0
    while hi - lo > tol:
        mid = round((lo + hi) / 2, digits)
        if mid in (lo, hi):
            break
        if fits(mid):
            hi = mid
        else:
            lo = mid
    return out  # out.ratio is hi: the last ratio that fitted


def flap_target_bytes(limit_bytes: int, agent_bytes: int) -> int:
    """The size FLAP is cut to when compared with a run's model: the run's memory limit, or the model's own size when
    the run ended over the limit (a one-layer-per-step run can), so both models are always the same size."""
    return max(limit_bytes, agent_bytes)


def flap_at_memory_limit(dense, stats, limit_bytes: int, hi: float = 0.6, tol: float = 0.001):
    """Baseline: FLAP: FLAP cut to a weight-memory limit (bytes of all parameters, as `memory`), at the smallest
    ratio that fits. `dense` is left unchanged; `stats` come from `flap.collect` on it. Returns (pruned model or
    None if the limit is out of reach, AtMemoryLimit)."""
    import copy

    from pruneagent import flap

    built: dict[float, object] = {}

    def weight_bytes(ratio: float) -> int:
        net = copy.deepcopy(dense)
        if ratio > 0:
            flap.prune_flap_from_stats(net, stats, ratio)
        built.clear()
        built[ratio] = net
        return memory(net)["weight_bytes"]

    try:
        out = smallest_fitting_ratio(weight_bytes, limit_bytes, hi=hi, tol=tol)
    except ValueError as e:  # e.g. the ratio would remove every attention group of a layer
        return None, AtMemoryLimit(limit_bytes, None, None, note=f"FLAP cannot prune at ratio {hi}: {e}")
    if out.ratio is None:
        return None, out
    if out.ratio not in built:
        weight_bytes(out.ratio)
    return built[out.ratio], out


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


def search_wanda50_match(
    ppl: Callable[[float], float], acc: Callable[[float], float], max_ppl: float, min_acc: float,
    lo: float = 0.0, hi: float = 0.5, tol: float = 0.005, acc_lo: float = 0.0, acc_tol: float = 0.01,
    acc_fails_at: float | None = None,
) -> Wanda50MatchSearch:
    """Bisect on perplexity, check accuracy at the ratio found; if it fails there, bisect on both parts over
    [acc_lo, that ratio] (to within `acc_tol`; acc_lo is evaluated if the search ends on it).
    `acc_fails_at`: a ratio already known to fail accuracy, so ratios at or above it are not checked again."""
    ratio, ppl_trials = bisect_ratio(ppl, max_ppl, lo, hi, tol)
    out = Wanda50MatchSearch(ratio=None, ppl_trials=ppl_trials)
    if not any(t.passed for t in ppl_trials):
        if ratio > 0:  # the search ended on lo, which was assumed to pass: check it
            v = ppl(ratio)
            out.ppl_trials.append(Trial(ratio, v, v <= max_ppl))
        if not any(t.passed for t in out.ppl_trials):
            out.note = f"no evaluated ratio keeps Wanda-50% perplexity (bisection stopped at {ratio})"
            return out

    def both(r: float) -> bool:
        v = ppl(r)
        out.ppl_trials.append(Trial(r, v, v <= max_ppl))
        if v > max_ppl:
            return False
        a = acc(r)
        out.acc_trials.append(Trial(r, a, a >= min_acc))
        return a >= min_acc

    if acc_fails_at is None or ratio < acc_fails_at:
        a = acc(ratio)
        out.acc_trials.append(Trial(ratio, a, a >= min_acc))
        if a >= min_acc:
            out.ratio, out.matches_wanda50 = ratio, True
            return out
        acc_hi = ratio
    else:
        acc_hi = acc_fails_at
    a_lo = acc_lo
    while acc_hi - a_lo > acc_tol:
        mid = round((a_lo + acc_hi) / 2, 4)
        if both(mid):
            a_lo = mid
        else:
            acc_hi = mid
    if a_lo > 0 and not any(t.ratio == a_lo and t.passed for t in out.acc_trials) and not both(a_lo):
        out.note = f"Wanda-50% accuracy not kept at any ratio tried down to {a_lo}"
        return out
    if a_lo == 0:
        out.note = "Wanda-50% accuracy not kept at any positive ratio tried"
        return out
    out.ratio, out.matches_wanda50 = a_lo, True
    return out


def wanda50_reference(wanda_ppl_json: Path, mc_json: Path, mc_run: str, ppl_margin: float = 1.0,
                      acc_margin: float = 0.0) -> dict:
    """The Wanda-50% reference (backup) from task 1.1's recorded results, and the limits a method must meet to
    keep its quality.
    `ppl_margin` multiplies the perplexity and `acc_margin` is subtracted from the accuracy; the defaults
    (1.0 and 0.0) mean no allowance. Task 1.9 used 1.10 and 0.02."""
    w = json.loads(Path(wanda_ppl_json).read_text())
    m = json.loads(Path(mc_json).read_text())["runs"][mc_run]
    return {
        "wanda50_validation_ppl": w["splits"]["validation"]["ppl"], "wanda50_test_ppl": w["splits"]["test"]["ppl"],
        "wanda50_mean_acc": m["mean_acc"], "ppl_margin": ppl_margin, "acc_margin": acc_margin,
        "max_validation_ppl": ppl_margin * w["splits"]["validation"]["ppl"],
        "min_mean_acc": m["mean_acc"] - acc_margin,
        "sources": {"ppl": str(wanda_ppl_json), "mc": f"{mc_json}#{mc_run}"},
    }


def memory(model) -> dict[str, int]:
    """Parameters and their bytes (tied embeddings counted once; compensation biases included)."""
    params = list(model.parameters())
    return {"parameters": sum(p.numel() for p in params),
            "weight_bytes": sum(p.numel() * p.element_size() for p in params)}
