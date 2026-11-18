"""Device target (ARCHITECTURE.md FR13, FR14; task 2.11): the early warning, the final warnings and the device report.

- Memory limit: weights only (the saved model file's size, `estimate.weight_bytes`), decided 2026-11-17. The
  key/value cache is reported beside it in the device report, at `estimate.REPORT_CONTEXT` tokens (2,048), and is
  not part of the limit.
- Early warning (`feasibility`, before any agent call): the limit is below the smallest model the rules allow
  (out of reach: the closest plan is that smallest model), or the unpruned model already fits. Either way the run
  goes straight to the final step. A minimum speed that even the smallest model misses is warned about too, but the
  search still runs for the memory limit.
- Stop: the search stops as soon as the current plan's weights fit (graph `stop_check`).
- Final warnings (`final_warnings`): the limit was not reached (closest plan and GB over), or the result keeps
  less quality than Baseline: FLAP on the gate's windows, or it misses the minimum speed; each with the tokens/s
  short of any minimum speed.
- Device report (`report`): fits or not, GB over, weight and cache sizes, the closest plan when the limit is not
  met, and estimated decode speed (with the per-layer cost) only for the M4 and the T4.

A minimum speed is accepted only for the M4 and the T4 (`RunConfig` refuses it elsewhere), so every speed gap here
comes from an estimate checked against measurements.
"""

from __future__ import annotations

import math

from pruneagent import estimate as E
from pruneagent import plan as P
from pruneagent.schema import DeviceReport, DeviceWarning, Plan, Rules, RunConfig

GB = E.GB


def smallest_plan(config, rules: Rules) -> Plan:
    """The smallest model the rules allow: every layer at its minimum MLP width and key/value groups, and
    `max_drops_per_run` layers dropped (embeddings unchanged). Unstructured sparsity frees no memory."""
    n_layers, width, _ = P.shape(config)
    min_mlp = math.ceil(width * rules.min_mlp_fraction)
    plan = P.dense_plan(config)
    for i in range(n_layers):
        while True:
            try:
                nxt = P.step(plan, i, "narrow_mlp", config)
            except P.PlanError:
                break
            if nxt[i]["mlp_keep"] < min_mlp:
                break
            plan = nxt
        plan[i]["groups_keep"] = min(rules.min_groups, plan[i]["groups_keep"])
    for i in range(min(rules.max_drops_per_run, n_layers - 1)):
        plan[i]["dropped"] = True
    return plan


def _speed_gap(config, run: RunConfig, plan: Plan) -> float:
    """Tokens/s short of the minimum speed (0 when none is set or it is met)."""
    if run.device.min_decode_tok_s is None:
        return 0.0
    return E.check(run.device, config, plan, layer_cost=run.layer_cost).speed_gap_tok_s


def _speed_text(config, run: RunConfig, plan: Plan) -> str:
    c = E.check(run.device, config, plan, layer_cost=run.layer_cost)
    return (f"its estimated decode speed is {c.decode_tok_s:.1f} tokens/s, {c.speed_gap_tok_s:.1f} short of the "
            f"minimum of {run.device.min_decode_tok_s:g}")


def feasibility(config, run: RunConfig) -> tuple[list[DeviceWarning], str | None]:
    """The early warning (FR13): (warnings, stop reason). A stop reason means the search is skipped."""
    limit, gb = run.limit_bytes, run.device.memory_limit_gb
    dense_plan = P.dense_plan(config)
    dense = E.weight_bytes(config, dense_plan)
    if dense <= limit:
        msg = (f"The unpruned model's weights ({dense / GB:.3f} GB) already fit the memory limit ({gb:g} GB): no "
               "pruning is needed.")
        gap = _speed_gap(config, run, dense_plan)
        if gap:
            msg += f" It is too slow, though: {_speed_text(config, run, dense_plan)}."
        return [DeviceWarning(kind="already_met", speed_gap_tok_s=gap, message=msg)], (
            "memory limit already met by the unpruned model")
    small = smallest_plan(config, run.rules)
    least = E.weight_bytes(config, small)
    gap_s = _speed_gap(config, run, small)
    if least > limit:
        gap = (least - limit) / GB
        msg = (f"The memory limit ({gb:g} GB) is out of reach: the smallest model the pruning rules allow still has "
               f"{least / GB:.3f} GB of weights, {gap:.3f} GB over the limit. The closest plan is that smallest model.")
        if gap_s:
            msg += f" It also misses the minimum speed: {_speed_text(config, run, small)}."
        return [DeviceWarning(kind="out_of_reach", closest_plan=small, memory_gap_gb=gap, speed_gap_tok_s=gap_s,
                              message=msg)], "memory limit out of reach"
    if gap_s:
        return [DeviceWarning(kind="out_of_reach", closest_plan=small, speed_gap_tok_s=gap_s, message=(
            f"The minimum speed is out of reach: even the smallest model the pruning rules allow is too slow; "
            f"{_speed_text(config, run, small)}. The search still runs for the memory limit."))], None
    return [], None


def final_warnings(config, run: RunConfig, plan: Plan, val_ppl: float, baseline_ppl: float) -> list[DeviceWarning]:
    """The final warnings (FR14) for the plan the search ended on, after a search."""
    size = E.weight_bytes(config, plan)
    gap_s = _speed_gap(config, run, plan)
    speed = f" It also misses the minimum speed: {_speed_text(config, run, plan)}." if gap_s else ""
    if size > run.limit_bytes:
        gap = (size - run.limit_bytes) / GB
        return [DeviceWarning(kind="out_of_reach", closest_plan=plan, memory_gap_gb=gap, speed_gap_tok_s=gap_s,
                              message=(f"The search ended without reaching the memory limit: the closest plan has "
                                       f"{size / GB:.3f} GB of weights, {gap:.3f} GB over the limit.{speed}"))]
    if val_ppl > baseline_ppl:
        return [DeviceWarning(kind="below_flap", closest_plan=plan, speed_gap_tok_s=gap_s, message=(
            f"The result fits but keeps less quality than Baseline: FLAP on the gate's windows (perplexity "
            f"{val_ppl:.2f} against {baseline_ppl:.2f}).{speed}"))]
    if gap_s:
        return [DeviceWarning(kind="out_of_reach", closest_plan=plan, speed_gap_tok_s=gap_s, message=(
            f"The result fits the memory limit but misses the minimum speed: {_speed_text(config, run, plan)}."))]
    return []


def report(config, run: RunConfig, plan: Plan, context: int = E.REPORT_CONTEXT) -> DeviceReport:
    """The device report for `plan` (the result, or the closest plan when the limit is out of reach)."""
    c = E.check(run.device, config, plan, layer_cost=run.layer_cost)
    w = c.estimate.weight_bytes
    speed = E.speed_machine(run.device) is not None
    return DeviceReport(
        target=run.device, fits_memory=c.fits_memory, memory_gap_gb=c.memory_gap_gb, weight_bytes=w,
        kv_cache_bytes=E.kv_cache_bytes(config, plan, context), kv_context=context,
        closest_plan=None if c.fits_memory else plan, speed_reported=speed,
        est_decode_tok_s=c.decode_tok_s if speed else None,
        speed_gap_tok_s=(c.speed_gap_tok_s if speed and run.device.min_decode_tok_s is not None else None))
