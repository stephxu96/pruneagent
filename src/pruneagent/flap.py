"""FLAP: fluctuation-based structured pruning (An et al., AAAI 2024), adapted to Qwen3.

Follows the reference code (github.com/CASIA-IVA-Lab/FLAP, commit 3bb57db, Apache-2.0) with its defaults:
WIFV metric, AL-AM structure (adaptive per layer and per module), bias compensation. Statistics are taken on the
unpruned model, block by block.

Calibration (one deliberate change from the reference): the shared C4 set every method here uses, 128 samples of
2,048 tokens (`data.flap_calib_name`), not the reference's 1,024 x 128 tokens of WikiText-2 train, which is also
the evaluation text's source. The statistics take any sample length: one update per sample, and scores are
standardised per layer, so the longer samples change no scale that matters. The WikiText-2 set remains only for
scripts/check_flap_reference.py (`data.flap_reference_calib_name`).

Adaptation to Qwen3's grouped-query attention: the reference removes single heads of 128 channels. Qwen3 shares
each key/value head between 2 query heads, so the unit removed here is a key/value group: one key/value head and
its query heads (2 x 128 = 256 channels of o_proj's input). A group's score is the mean of its channels' scores,
as the reference does for a head. In the global allocation a group counts as many MLP neurons as it has
parameters (reference: 512/3 for a LLaMA head; Qwen3-0.6B: 256).

Per channel j of the input X to o_proj (attention) or down_proj (MLP):
    WIFV_j = fluctuation(X_j) * sum_i W[i, j]^2      (attention channels use WIFV_j squared, as the reference)
Each layer's scores are standardised (mean 0, standard deviation 1) so layers compare; one global threshold then
decides what every layer keeps. Removed channels' average contribution, W[:, removed] @ mean(X[removed]), is added
to the output projection as a bias so the layer's average output is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from pruneagent.wanda import _block_inputs, _run_block


@dataclass
class FlapStats:
    """Per-block input statistics of o_proj and down_proj on the unpruned model (float32, CPU)."""

    attn_fluct: torch.Tensor  # [layers, q_dim]
    attn_mean: torch.Tensor  # [layers, q_dim]
    mlp_fluct: torch.Tensor  # [layers, intermediate]
    mlp_mean: torch.Tensor  # [layers, intermediate]


class _Fluct:
    """Running mean and fluctuation of a linear layer's input, updated exactly as the reference `BiasGPT`
    (WIFV branch): one update per calibration sample; the first sample sets the mean but adds no fluctuation;
    the reference's 'old mean' is an alias of the updated mean, so both factors use the new mean."""

    def __init__(self, dim: int, device: str):
        self.mean = torch.zeros(dim, device=device)
        self.fluct = torch.zeros(dim, device=device)
        self.n = 0

    def add(self, x: torch.Tensor) -> None:
        x = x.detach().float().reshape(-1, x.shape[-1]).t()  # (dim, tokens), one sample
        self.mean *= self.n / (self.n + 1)
        self.mean += x.mean(dim=1) / (self.n + 1)
        if self.n == 0:
            self.fluct.zero_()
        else:
            self.fluct *= (self.n - 1) / self.n
            self.fluct += ((x - self.mean.unsqueeze(1)) ** 2).sum(dim=1) / (self.n + 1)
        self.n += 1


@torch.no_grad()
def collect(model, calib: torch.Tensor, device: str, log=None) -> FlapStats:
    """Feed the calibration samples through the unpruned model block by block, recording o_proj and down_proj
    input statistics. Nothing is pruned here."""
    hiddens, kwargs = _block_inputs(model, calib, device)
    out: dict[str, list[torch.Tensor]] = {"af": [], "am": [], "mf": [], "mm": []}
    for i, block in enumerate(model.model.layers):
        o, d = block.self_attn.o_proj, block.mlp.down_proj
        acc = {"attn": _Fluct(o.in_features, device), "mlp": _Fluct(d.in_features, device)}
        handles = [
            o.register_forward_hook(lambda _m, a, _o, f=acc["attn"]: f.add(a[0])),
            d.register_forward_hook(lambda _m, a, _o, f=acc["mlp"]: f.add(a[0])),
        ]
        hiddens = [_run_block(block, h, kwargs) for h in hiddens]
        for h in handles:
            h.remove()
        out["af"].append(acc["attn"].fluct.cpu())
        out["am"].append(acc["attn"].mean.cpu())
        out["mf"].append(acc["mlp"].fluct.cpu())
        out["mm"].append(acc["mlp"].mean.cpu())
        if log:
            log(f"  block {i:2d}: statistics collected")
    return FlapStats(*(torch.stack(out[k]) for k in ("af", "am", "mf", "mm")))


def group_shape(model) -> tuple[int, int]:
    """(number of key/value groups, o_proj input channels per group)."""
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return cfg.num_key_value_heads, head_dim * cfg.num_attention_heads // cfg.num_key_value_heads


def _standardize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True)


@torch.no_grad()
def scores(model, stats: FlapStats) -> tuple[torch.Tensor, torch.Tensor]:
    """Standardised WIFV scores: attention [layers, groups] and MLP [layers, intermediate]. Higher = keep."""
    o_sq = torch.stack([b.self_attn.o_proj.weight.float().pow(2).sum(dim=0).cpu() for b in model.model.layers])
    d_sq = torch.stack([b.mlp.down_proj.weight.float().pow(2).sum(dim=0).cpu() for b in model.model.layers])
    n_groups, per_group = group_shape(model)
    attn = _standardize((stats.attn_fluct * o_sq) ** 2)
    attn = attn.reshape(attn.shape[0], n_groups, per_group).mean(dim=2)
    mlp = _standardize(stats.mlp_fluct * d_sq)
    return attn, mlp


def group_weight(model) -> float:
    """Parameters in one key/value group (q, k, v rows and o columns) divided by parameters in one MLP neuron."""
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    q_heads = cfg.num_attention_heads // cfg.num_key_value_heads
    group_params = (2 * q_heads + 2) * head_dim * cfg.hidden_size
    return group_params / (3 * cfg.hidden_size)


def allocate(
    attn: torch.Tensor, mlp: torch.Tensor, ratio: float, weight: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """AL-AM: one global threshold over all groups and neurons so that about `ratio` of block parameters go.
    Returns keep-masks [layers, groups] and [layers, intermediate]. Reference code, adapted weight."""
    metric = torch.cat([attn.reshape(-1), mlp.reshape(-1)])
    sorted_metric, indices = torch.sort(metric, descending=True)
    cost = torch.ones_like(sorted_metric)
    cost[indices < attn.numel()] = weight
    target = cost.sum() * (1 - ratio)
    threshold = sorted_metric[torch.argmin(torch.abs(torch.cumsum(cost, 0) - target))]
    return attn > threshold, mlp > threshold


def _bias_add(linear: nn.Linear, extra: torch.Tensor) -> None:
    extra = extra.to(linear.weight.dtype)
    if linear.bias is None:
        linear.bias = nn.Parameter(extra.clone(), requires_grad=False)
    else:
        linear.bias.data += extra


@torch.no_grad()
def remove_mlp_neurons(block, keep: torch.Tensor, mean: torch.Tensor | None) -> None:
    """Physically remove MLP neurons where `keep` is False; with `mean`, compensate with a down_proj bias."""
    mlp = block.mlp
    dev = mlp.down_proj.weight.device
    keep = keep.to(dev)
    idx = torch.where(keep)[0]
    if mean is not None:
        w = mlp.down_proj.weight.float()
        _bias_add(mlp.down_proj, (mean.to(dev).float() * ~keep) @ w.T)
    for lin in (mlp.gate_proj, mlp.up_proj):
        lin.weight = nn.Parameter(lin.weight[idx].contiguous(), requires_grad=False)
        lin.out_features = len(idx)
    mlp.down_proj.weight = nn.Parameter(mlp.down_proj.weight[:, idx].contiguous(), requires_grad=False)
    mlp.down_proj.in_features = len(idx)
    if hasattr(mlp, "intermediate_size"):
        mlp.intermediate_size = len(idx)


@torch.no_grad()
def remove_attention_groups(block, keep_groups: torch.Tensor, mean: torch.Tensor | None) -> None:
    """Physically remove key/value groups (each key/value head and its query heads) where `keep_groups` is False;
    with `mean` (o_proj input mean over current channels), compensate with an o_proj bias."""
    attn = block.self_attn
    if not keep_groups.any():
        raise ValueError("cannot remove every attention group; drop the layer instead")
    head_dim = attn.head_dim
    n_groups = attn.k_proj.out_features // head_dim
    q_per_group = attn.q_proj.out_features // head_dim // n_groups
    dev = attn.o_proj.weight.device
    keep_groups = keep_groups.to(dev)
    q_keep = keep_groups.repeat_interleave(q_per_group * head_dim)
    kv_keep = keep_groups.repeat_interleave(head_dim)
    if mean is not None:
        w = attn.o_proj.weight.float()
        _bias_add(attn.o_proj, (mean.to(dev).float() * ~q_keep) @ w.T)
    q_idx, kv_idx = torch.where(q_keep)[0], torch.where(kv_keep)[0]
    for lin, idx in ((attn.q_proj, q_idx), (attn.k_proj, kv_idx), (attn.v_proj, kv_idx)):
        lin.weight = nn.Parameter(lin.weight[idx].contiguous(), requires_grad=False)
        if lin.bias is not None:
            lin.bias = nn.Parameter(lin.bias[idx].contiguous(), requires_grad=False)
        lin.out_features = len(idx)
    attn.o_proj.weight = nn.Parameter(attn.o_proj.weight[:, q_idx].contiguous(), requires_grad=False)
    attn.o_proj.in_features = len(q_idx)


@torch.no_grad()
def prune_flap(model, calib: torch.Tensor, ratio: float, device: str, log=None) -> dict:
    """Prune `model` in place with FLAP (WIFV, AL-AM, bias compensation) at `ratio` of block parameters.
    Returns the keep-masks and the statistics."""
    return prune_flap_from_stats(model, collect(model, calib, device, log=log), ratio)


@torch.no_grad()
def prune_flap_from_stats(model, stats: FlapStats, ratio: float) -> dict:
    """As `prune_flap`, with statistics already collected on this (unpruned) model. Only the global threshold
    depends on `ratio`, so one `collect` serves every ratio."""
    attn, mlp = scores(model, stats)
    attn_keep, mlp_keep = allocate(attn, mlp, ratio, group_weight(model))
    for i, block in enumerate(model.model.layers):
        remove_attention_groups(block, attn_keep[i], stats.attn_mean[i])
        remove_mlp_neurons(block, mlp_keep[i], stats.mlp_mean[i])
    return {"attn_keep": attn_keep, "mlp_keep": mlp_keep, "stats": stats}


def block_params(model) -> int:
    """Weights in all blocks' linear layers (biases excluded), for measuring how much was removed."""
    return sum(
        lin.weight.numel()
        for b in model.model.layers
        for lin in (b.self_attn.q_proj, b.self_attn.k_proj, b.self_attn.v_proj, b.self_attn.o_proj,
                    b.mlp.gate_proj, b.mlp.up_proj, b.mlp.down_proj)
    )
