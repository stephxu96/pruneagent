"""Estimate-vs-measured speed check (task 1.7; assumptions A6 and A8).

For each plan and each measure (decode tokens/s, prefill tokens/s) the estimate (`estimate.estimate` against a
device target) is set beside the speed meter's median (`speed.measure`). Reported per measure:

    spearman             rank correlation between estimated and measured speed over the plans (A6: >= 0.7)
    error                (estimate - measured) / measured for each plan; positive = estimate too fast
    mean_abs_error       mean of |error| over the plans (A8: within ~0.25)
    max_abs_error        largest |error|

Two optional, separately labelled corrections, each fitted by least squares on the same measurements it is then
scored on (in-sample, so they flatter the estimate) and never used for the A6 / A8 numbers:
`fit_overhead`  measured time per token = estimated + a fixed overhead;
`fit_time`      measured time per token = scale x estimated + a fixed overhead.
"""

from __future__ import annotations

import statistics

MEASURES = ("decode", "prefill")


def ranks(values: list[float]) -> list[float]:
    """1-based ranks; tied values share the mean of their ranks."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y):
        raise ValueError("x and y differ in length")
    if len(x) < 2:
        return None
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / (sxx * syy) ** 0.5 if sxx and syy else None


def spearman(x: list[float], y: list[float]) -> float | None:
    """Spearman rank correlation (Pearson correlation of the ranks, ties averaged). None if either side is constant."""
    return pearson(ranks(x), ranks(y))


def errors(estimated: list[float], measured: list[float]) -> list[float]:
    return [(e - m) / m for e, m in zip(estimated, measured, strict=True)]


def score(estimated: list[float], measured: list[float]) -> dict:
    err = errors(estimated, measured)
    return {"spearman": spearman(estimated, measured), "error": err,
            "mean_abs_error": statistics.fmean(abs(e) for e in err), "max_abs_error": max(abs(e) for e in err),
            "mean_error": statistics.fmean(err)}


def fit_time(estimated: list[float], measured: list[float]) -> dict:
    """Least-squares line measured_time = scale x estimated_time + overhead, on seconds per token (1 / speed).
    Returns the fit and the corrected speeds 1 / (scale / estimated + overhead)."""
    te, tm = [1 / v for v in estimated], [1 / v for v in measured]
    me, mm = statistics.fmean(te), statistics.fmean(tm)
    sxx = sum((a - me) ** 2 for a in te)
    scale = sum((a - me) * (b - mm) for a, b in zip(te, tm, strict=True)) / sxx if sxx else 0.0
    overhead = mm - scale * me
    corrected = [1 / (scale * t + overhead) for t in te]
    return {"scale": scale, "overhead_s_per_token": overhead, "corrected": corrected,
            **{k: v for k, v in score(corrected, measured).items() if k != "spearman"},
            "spearman": spearman(corrected, measured)}


def fit_overhead(estimated: list[float], measured: list[float]) -> dict:
    """The simplest correction: a fixed overhead added to every token's estimated time (least squares, so the
    mean of measured minus estimated seconds per token). Corrected speed = 1 / (1 / estimated + overhead)."""
    overhead = statistics.fmean(1 / m - 1 / e for e, m in zip(estimated, measured, strict=True))
    corrected = [1 / (1 / e + overhead) for e in estimated]
    return {"overhead_s_per_token": overhead, "corrected": corrected,
            **{k: v for k, v in score(corrected, measured).items() if k != "spearman"},
            "spearman": spearman(corrected, measured)}


def compare(rows: list[dict], targets: list[str]) -> dict:
    """`rows`: one per plan, {"plan", "measured": {"decode", "prefill"}, "estimated": {target: {"decode",
    "prefill"}}}. Returns {target: {measure: score(...) + {"fit": fit_time(...), "fit_overhead": ...}}}."""
    out: dict = {}
    for t in targets:
        out[t] = {}
        for m in MEASURES:
            est = [r["estimated"][t][m] for r in rows]
            meas = [r["measured"][m] for r in rows]
            out[t][m] = score(est, meas) | {"fit": fit_time(est, meas), "fit_overhead": fit_overhead(est, meas)}
    return out
