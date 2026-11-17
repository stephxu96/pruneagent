"""Layer probes (ARCHITECTURE.md §5, FR2): how much one layer's output changes when a strategy is applied to it.

`capture` runs the unpruned model once over a few calibration samples and keeps every block's input (and the
last block's output). `probe` then works on one layer at a time from those cached activations: it builds that
layer's block for the plan with 1, 2 or 3 steps of a strategy applied (`plan.step`, `pruner.prune_block`, so
the same layer the pruner builds), runs it on the cached inputs and measures

    local_error = ||block_pruned(x) - block_dense(x)||^2 / ||block_dense(x)||^2   (summed over all tokens)

where x is the unpruned model's input to that block and a dropped layer becomes the identity (output = x).
Each probe also carries the layer's Block Influence (ShortGPT): 1 - mean cosine similarity of block input and
output over tokens.

Choices where the architecture is silent:
- Calibration: the first 8 of the cached C4 calibration samples (8 x 2,048 tokens), not evaluation data.
- Inputs always come from the unpruned model, even when probing ahead of a pruned plan; earlier layers' pruning
  is ignored, so a probe measures the layer alone.
"""

from __future__ import annotations

import copy
from typing import TypedDict

import torch

from pruneagent import flap
from pruneagent import plan as P
from pruneagent.pruner import Scores, prune_block
from pruneagent.wanda import _block_inputs, _run_block

PROBE_SAMPLES = 8


class Candidate(TypedDict):
    strategy: str
    steps: int
    setting: P.LayerPlan | None  # the layer's plan after the steps; None when the steps are impossible
    local_error: float | None


class LayerProbe(TypedDict):
    layer: int
    block_influence: float
    candidates: list[Candidate]


class Activations:
    """Unpruned-model block inputs on calibration samples: `inputs[i]` feeds block i, `inputs[n]` is the last
    block's output. Each entry is a list of [1, seqlen, hidden] tensors, one per sample."""

    def __init__(self, inputs: list[list[torch.Tensor]], kwargs: dict):
        self.inputs = inputs
        self.kwargs = kwargs

    def block_in(self, i: int) -> list[torch.Tensor]:
        return self.inputs[i]

    def block_out(self, i: int) -> list[torch.Tensor]:
        return self.inputs[i + 1]


@torch.no_grad()
def capture(model, calib: torch.Tensor, device: str) -> Activations:
    """One pass of `calib` ([samples, seqlen] token ids) through the unpruned model, keeping each block's input."""
    hiddens, kwargs = _block_inputs(model, calib, device)
    inputs = [hiddens]
    for block in model.model.layers:
        hiddens = [_run_block(block, h, kwargs) for h in hiddens]
        inputs.append(hiddens)
    return Activations(inputs, kwargs)


def relative_error(out: list[torch.Tensor], ref: list[torch.Tensor]) -> float:
    num = sum(float((o.float() - r.float()).pow(2).sum()) for o, r in zip(out, ref, strict=True))
    den = sum(float(r.float().pow(2).sum()) for r in ref)
    return num / den


def block_influence(x: list[torch.Tensor], y: list[torch.Tensor]) -> float:
    cos = torch.cat([torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1).reshape(-1)
                     for a, b in zip(x, y, strict=True)])
    return float(1 - cos.mean())


@torch.no_grad()
def probe(model, layer: int, strategies, acts: Activations, scores: Scores, plan: P.Plan | None = None,
          steps=(1,)) -> LayerProbe:
    """Local error of `layer` for each strategy at each number of steps ahead of `plan` (default: unpruned).
    `model` must be the unpruned model; it is not changed."""
    cfg = model.config
    base = P.normalize(plan or {}, cfg)
    n_groups, _ = flap.group_shape(model)
    width = cfg.intermediate_size
    x, ref = acts.block_in(layer), acts.block_out(layer)
    out = LayerProbe(layer=layer, block_influence=block_influence(x, ref), candidates=[])
    for strategy in strategies:
        for k in steps:
            try:
                new = P.apply_steps(base, [{"layer": layer, "strategy": strategy}] * k, cfg)
            except P.PlanError:
                out["candidates"].append(Candidate(strategy=strategy, steps=k, setting=None, local_error=None))
                continue
            setting = new[layer]
            if setting["dropped"]:
                y = x
            else:
                block = copy.deepcopy(model.model.layers[layer])
                prune_block(block, layer, setting, scores, n_groups, width)
                y = [_run_block(block, h, acts.kwargs) for h in x]
                del block
            out["candidates"].append(Candidate(strategy=strategy, steps=k, setting=dict(setting),
                                               local_error=relative_error(y, ref)))
    return out


def sensitivity_table(model, acts: Activations, scores: Scores, strategies=P.STRATEGIES, steps=(1,),
                      layers=None, plan: P.Plan | None = None) -> dict[int, LayerProbe]:
    layers = range(len(model.model.layers)) if layers is None else layers
    return {i: probe(model, i, strategies, acts, scores, plan, steps) for i in layers}


def merge_probes(a: dict[int, LayerProbe] | None, b: dict[int, LayerProbe] | None) -> dict[int, LayerProbe]:
    """Reducer for parallel probes (§4.3): later results for a layer replace earlier ones."""
    return {**(a or {}), **(b or {})}


def _ranks(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for t in range(i, j + 1):
            r[order[t]] = (i + j) / 2  # ties share the average rank
        i = j + 1
    return r


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation (Pearson correlation of average ranks)."""
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("need two equal-length lists of at least 2 values")
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    return cov / (vx * vy) ** 0.5 if vx and vy else float("nan")


def compare(table: dict[int, LayerProbe], ppl_dense: float, ppl: dict[tuple[int, str], float]) -> dict:
    """Spearman correlation of one-step local error with the full-model perplexity increase, per strategy
    (ranking layers) and over all (layer, strategy) pairs; plus each strategy's largest rank disagreements."""
    rows = []
    for i, lp in sorted(table.items()):
        for c in lp["candidates"]:
            if c["steps"] == 1 and c["local_error"] is not None and (i, c["strategy"]) in ppl:
                rows.append((i, c["strategy"], c["local_error"], ppl[i, c["strategy"]] - ppl_dense,
                             lp["block_influence"]))
    out: dict = {"n": len(rows), "per_strategy": {}}
    if len(rows) >= 2:
        out["all"] = spearman([r[2] for r in rows], [r[3] for r in rows])
    for s in sorted({r[1] for r in rows}):
        sub = [r for r in rows if r[1] == s]
        if len(sub) < 2:
            continue
        rp, rt = _ranks([r[2] for r in sub]), _ranks([r[3] for r in sub])
        gaps = sorted(range(len(sub)), key=lambda k: -abs(rp[k] - rt[k]))[:3]
        out["per_strategy"][s] = {
            "n": len(sub), "spearman": spearman([r[2] for r in sub], [r[3] for r in sub]),
            "spearman_block_influence": spearman([r[4] for r in sub], [r[3] for r in sub]),
            "largest_rank_gaps": [{"layer": sub[k][0], "probe_rank": rp[k], "ppl_rank": rt[k]} for k in gaps],
        }
    return out
