"""Wanda-sp: structured Wanda, the uniform baseline introduced with FLAP (An et al., AAAI 2024), adapted to Qwen3.

Follows `prune_wanda_sp` in the reference code (github.com/CASIA-IVA-Lab/FLAP, commit 3bb57db, Apache-2.0).
Block by block, on inputs that have passed through the already-pruned earlier blocks:

1. Hooks on o_proj and down_proj accumulate `scaler_row`, the mean over calibration samples of each input
   channel's squared norm (reference `WrappedGPT`).
2. Score of weight W[i, j] = |W[i, j]| * sqrt(scaler_row[j]). Attention: mean over output rows, then summed over
   each unit's channels. MLP: mean over output rows, one score per neuron.
3. Every block loses the same share: threshold = sort(scores)[int(ratio * units)]; units scoring >= threshold
   stay. The block is pruned without bias compensation, then rerun to give the next block its inputs.

Deviations from the reference, one line each:
- Attention unit is a Qwen3 key/value group (1 key/value head + its 2 query heads, 256 o_proj channels), not a
  128-channel head; Qwen3-0.6B has 8 groups per layer, so int(0.2 * 8) = 1 group goes per layer at 20%.
- Calibration loops over the 128 real samples; the reference loops over --nsamples (1,024 or 2,048) rows of which
  all but 128 are zeros. Zero rows give zero o_proj/down_proj inputs, so they only rescale scaler_row by a constant
  and cannot change which units are kept.
- Calibration samples are the shared C4 set every method here uses (128 x 2,048 tokens, `data.wanda_sp_calib_name`),
  not the reference's 128 x 128-token C4 windows; those remain only for scripts/check_wanda_sp_reference.py.
"""

from __future__ import annotations

import torch

from pruneagent.flap import group_shape, remove_attention_groups, remove_mlp_neurons
from pruneagent.wanda import _block_inputs, _run_block


class _ScalerRow:
    """Running mean of each input channel's squared norm, updated exactly as the reference `WrappedGPT.add_batch`
    (one update per calibration sample)."""

    def __init__(self, dim: int, device: str):
        self.scaler_row = torch.zeros(dim, device=device)
        self.nsamples = 0

    def add(self, x: torch.Tensor) -> None:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        batch = x.shape[0]
        x = x.detach().reshape(-1, x.shape[-1]).t().float()
        self.scaler_row *= self.nsamples / (self.nsamples + batch)
        self.nsamples += batch
        self.scaler_row += torch.norm(x, p=2, dim=1) ** 2 / self.nsamples


def attention_scores(o_weight: torch.Tensor, scaler_row: torch.Tensor, per_group: int) -> torch.Tensor:
    """One score per key/value group: |W| * sqrt(scaler_row), mean over output rows, summed per group."""
    metric = o_weight.abs().float() * torch.sqrt(scaler_row.float()).reshape(1, -1)
    return metric.mean(dim=0).reshape(-1, per_group).sum(dim=1)


def mlp_scores(down_weight: torch.Tensor, scaler_row: torch.Tensor) -> torch.Tensor:
    """One score per MLP neuron: |W| * sqrt(scaler_row), mean over output rows."""
    return (down_weight.abs().float() * torch.sqrt(scaler_row.float()).reshape(1, -1)).mean(dim=0)


def keep_mask(metric: torch.Tensor, ratio: float) -> torch.Tensor:
    """Reference threshold: the int(ratio * n)-th smallest score; units scoring at least that stay."""
    thresh = torch.sort(metric)[0][int(ratio * metric.numel())]
    return metric >= thresh


@torch.no_grad()
def prune_wanda_sp(model, calib: torch.Tensor, ratio: float, device: str, log=None) -> dict:
    """Prune `model` in place with uniform Wanda-sp, removing `ratio` of each block's attention groups and MLP
    neurons. Returns keep-masks: attention [layers, groups] and MLP [layers, intermediate]."""
    if not 0 <= ratio < 1:
        raise ValueError(f"ratio must be in [0, 1), got {ratio}")
    use_cache = model.config.use_cache
    model.config.use_cache = False  # as the reference; a key/value cache would break once groups are removed
    try:
        return _prune(model, calib, ratio, device, log)
    finally:
        model.config.use_cache = use_cache


def _prune(model, calib: torch.Tensor, ratio: float, device: str, log) -> dict:
    _, per_group = group_shape(model)
    hiddens, kwargs = _block_inputs(model, calib, device)
    attn_keep, mlp_keep = [], []
    for i, block in enumerate(model.model.layers):
        o, d = block.self_attn.o_proj, block.mlp.down_proj
        acc = {"attn": _ScalerRow(o.in_features, device), "mlp": _ScalerRow(d.in_features, device)}
        handles = [
            o.register_forward_hook(lambda _m, a, _o, s=acc["attn"]: s.add(a[0])),
            d.register_forward_hook(lambda _m, a, _o, s=acc["mlp"]: s.add(a[0])),
        ]
        for h in hiddens:
            _run_block(block, h, kwargs)
        for h in handles:
            h.remove()
        a_keep = keep_mask(attention_scores(o.weight, acc["attn"].scaler_row, per_group), ratio)
        m_keep = keep_mask(mlp_scores(d.weight, acc["mlp"].scaler_row), ratio)
        remove_attention_groups(block, a_keep, None)
        remove_mlp_neurons(block, m_keep, None)
        attn_keep.append(a_keep.cpu())
        mlp_keep.append(m_keep.cpu())
        hiddens = [_run_block(block, h, kwargs) for h in hiddens]
        if log:
            log(f"  block {i:2d}: groups kept {int(a_keep.sum())}, neurons kept {int(m_keep.sum())}")
    return {"attn_keep": torch.stack(attn_keep), "mlp_keep": torch.stack(mlp_keep)}
