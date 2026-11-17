"""Pruner: build the pruned model for a per-layer plan from the unpruned model and cached scores (ADR-002).

Scores are computed once on the unpruned model (`compute_scores`) and saved; pruning from them is slicing and
masking only, with no calibration pass, so the same plan always gives the same model.

Per kept layer, in this order:
1. Attention: keep the `groups_keep` key/value groups with the highest FLAP score (ties: lower index first);
   removed groups' average contribution is added to o_proj as a bias (FLAP bias compensation).
2. MLP: keep the `mlp_keep` neurons with the highest FLAP score, same tie rule; compensation bias on down_proj.
3. Wanda: in each of the seven linear layers, zero int(remaining inputs x sparsity) weights per output row
   with the lowest |W| x ||X||, X taken at the remaining input channels. Biases are not pruned.
Then dropped layers are removed and the remaining layers renumbered (attention layer_idx, config
num_hidden_layers and layer_types), so generation with a key/value cache works.

Choices where the architecture is silent:
- Order: structured cuts before Wanda, so sparsity applies to the remaining weights (§4.2); compensation uses
  only removed columns, which Wanda never touches, so the order does not change the bias.
- Wanda's activation norms come from one pass of the C4 calibration data through the unpruned model, not from
  the structurally pruned one. Deviation from published Wanda, which calibrates layer by layer on the pruned
  earlier layers: cached unpruned-model scores are what ADR-002 prescribes so a rebuild needs no calibration.
- Dropped layers are skipped entirely (their other plan fields are ignored).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from pruneagent import flap
from pruneagent import plan as P
from pruneagent.wanda import LINEARS, _block_inputs, _run_block, _sublayer, row_mask


@dataclass
class Scores:
    """Everything pruning needs, from the unpruned model (float32, CPU)."""

    stats: flap.FlapStats  # o_proj / down_proj input means (for bias compensation) and fluctuations
    attn: torch.Tensor  # [layers, groups] standardised FLAP scores, higher = keep
    mlp: torch.Tensor  # [layers, intermediate]
    wanda_sq: dict[str, torch.Tensor]  # linear name -> [layers, in_features]: sum of squared inputs
    meta: dict[str, str] = field(default_factory=dict)  # provenance (model, calibration sets)


@torch.no_grad()
def collect_wanda_norms(model, calib: torch.Tensor, device: str, log=None) -> dict[str, torch.Tensor]:
    """Sum of squared inputs to each block linear layer over all calibration tokens, on the unpruned model."""
    hiddens, kwargs = _block_inputs(model, calib, device)
    out: dict[str, list[torch.Tensor]] = {name: [] for name in LINEARS}
    for i, block in enumerate(model.model.layers):
        acc = {name: torch.zeros(_sublayer(block, name).in_features, device=device) for name in LINEARS}

        def make_hook(a: torch.Tensor):
            def hook(_m, args, _o):
                a.add_(args[0].detach().float().reshape(-1, args[0].shape[-1]).pow(2).sum(dim=0))
            return hook

        handles = [_sublayer(block, n).register_forward_hook(make_hook(acc[n])) for n in LINEARS]
        hiddens = [_run_block(block, h, kwargs) for h in hiddens]
        for h in handles:
            h.remove()
        for n in LINEARS:
            out[n].append(acc[n].cpu())
        if log:
            log(f"  block {i:2d}: Wanda input norms collected")
    return {n: torch.stack(v) for n, v in out.items()}


def compute_scores(model, flap_calib: torch.Tensor, wanda_calib: torch.Tensor, device: str,
                   meta: dict[str, str] | None = None, log=None) -> Scores:
    """FLAP statistics and scores (FLAP's WikiText-2 calibration) and Wanda input norms (C4 calibration),
    all on the unpruned model. The model is not changed."""
    stats = flap.collect(model, flap_calib, device, log=log)
    attn, mlp = flap.scores(model, stats)
    wanda_sq = collect_wanda_norms(model, wanda_calib, device, log=log)
    return Scores(stats=stats, attn=attn, mlp=mlp, wanda_sq=wanda_sq, meta=dict(meta or {}))


def save_scores(scores: Scores, path) -> None:
    s = scores.stats
    tensors = {"attn_fluct": s.attn_fluct, "attn_mean": s.attn_mean, "mlp_fluct": s.mlp_fluct,
               "mlp_mean": s.mlp_mean, "attn": scores.attn, "mlp": scores.mlp}
    tensors |= {f"wanda_sq.{n}": t for n, t in scores.wanda_sq.items()}
    save_file({k: v.contiguous().float().cpu() for k, v in tensors.items()}, str(path),
              metadata={k: str(v) for k, v in scores.meta.items()})


def load_scores(path) -> Scores:
    from safetensors import safe_open

    t = load_file(str(path))
    with safe_open(str(path), "pt") as f:
        meta = f.metadata() or {}
    stats = flap.FlapStats(t["attn_fluct"], t["attn_mean"], t["mlp_fluct"], t["mlp_mean"])
    wanda_sq = {n: t[f"wanda_sq.{n}"] for n in LINEARS}
    return Scores(stats=stats, attn=t["attn"], mlp=t["mlp"], wanda_sq=wanda_sq, meta=meta)


def top_mask(score: torch.Tensor, k: int) -> torch.Tensor:
    """Boolean mask of the k highest scores; ties go to the lower index (stable, deterministic)."""
    order = torch.sort(score.float(), descending=True, stable=True).indices
    keep = torch.zeros_like(score, dtype=torch.bool)
    keep[order[:k]] = True
    return keep


def _check_dense(model) -> None:
    cfg = model.config
    n_layers, width, groups = P.shape(cfg)
    _, per_group = flap.group_shape(model)
    if len(model.model.layers) != n_layers or any(
        b.mlp.down_proj.in_features != width or b.self_attn.o_proj.in_features != groups * per_group
        for b in model.model.layers
    ):
        raise ValueError("apply_plan needs the unpruned model; rebuild from a dense copy (see `prune`)")


def drop_layers(model, dropped: list[int]) -> None:
    """Remove the given layers and renumber the rest so the key/value cache indexes them correctly."""
    if not dropped:
        return
    cfg = model.config
    keep = [i for i in range(len(model.model.layers)) if i not in set(dropped)]
    types = getattr(cfg, "layer_types", None)
    model.model.layers = nn.ModuleList([model.model.layers[i] for i in keep])
    for new_idx, block in enumerate(model.model.layers):
        block.self_attn.layer_idx = new_idx
    cfg.num_hidden_layers = len(keep)
    if types is not None:
        cfg.layer_types = [types[i] for i in keep]


@torch.no_grad()
def apply_plan(model, plan: dict, scores: Scores):
    """Prune the unpruned `model` in place to `plan` (validated first) and return it. See module docstring."""
    full = P.validate(plan, model.config)
    _check_dense(model)
    dense_config = model.config.to_dict()
    n_groups, _ = flap.group_shape(model)
    width = model.config.intermediate_size
    for i, block in enumerate(model.model.layers):
        p = full[i]
        if p["dropped"]:
            continue
        in_idx = {}
        if p["groups_keep"] < n_groups:
            keep = top_mask(scores.attn[i], p["groups_keep"])
            flap.remove_attention_groups(block, keep, scores.stats.attn_mean[i])
            per = block.self_attn.head_dim * block.self_attn.num_key_value_groups
            in_idx["self_attn.o_proj"] = torch.where(keep.repeat_interleave(per))[0]
        if p["mlp_keep"] < width:
            keep = top_mask(scores.mlp[i], p["mlp_keep"])
            flap.remove_mlp_neurons(block, keep, scores.stats.mlp_mean[i])
            in_idx["mlp.down_proj"] = torch.where(keep)[0]
        if p["wanda_sparsity"] > 0:
            for name in LINEARS:
                lin = _sublayer(block, name)
                sq = scores.wanda_sq[name][i]
                if name in in_idx:
                    sq = sq[in_idx[name]]
                mask = row_mask(lin.weight, sq.to(lin.weight.device), p["wanda_sparsity"])
                lin.weight.mul_(mask.to(lin.weight.dtype))
    drop_layers(model, [i for i, p in full.items() if p["dropped"]])
    model.pruneagent = {"dense_config": dense_config, "plan": full}
    return model


def prune(dense, plan: dict, scores: Scores):
    """A pruned copy of `dense` (left unchanged)."""
    return apply_plan(copy.deepcopy(dense), plan, scores)


def weight_counts(model) -> dict[str, int]:
    """Block linear weights present, of which zero (from Wanda), and all parameters."""
    present = flap.block_params(model)
    zeros = sum(int((_sublayer(b, n).weight == 0).sum()) for b in model.model.layers for n in LINEARS)
    return {"block_weights": present, "block_zeros": zeros,
            "parameters": sum(p.numel() for p in model.parameters())}
