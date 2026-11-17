"""Wanda: pruning by weights and activations (Sun et al., ICLR 2024), adapted to Qwen3.

Scoring and per-row masking are adapted from Steph Xu's Lab 2 (`lab2_pruning_steph.ipynb`, `WandaPruner`),
which already took a per-layer sparsity list. Three changes bring it in line with the reference code
(github.com/locuslab/wanda) so the quality bar matches published Wanda:

1. All seven linear layers in each block are pruned (attention q/k/v/o and MLP gate/up/down), not only the MLP.
2. Calibration uses the cached C4 samples (128 x 2,048 tokens), not WikiText test tokens.
3. Layers are calibrated in order: each layer is scored on inputs that have already passed through the pruned
   earlier layers, then rerun after pruning to produce the next layer's inputs.

Score for weight W[i, j] = |W[i, j]| * ||X_j||_2, where X_j is input feature j over all calibration tokens.
In each output row, the int(in_features * sparsity) lowest-scoring weights are set to zero.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

LINEARS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def row_mask(weight: torch.Tensor, act_sq_sum: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Boolean keep-mask: in every output row, drop the int(in_features * sparsity) lowest |W| * ||X|| scores."""
    keep = torch.ones_like(weight, dtype=torch.bool)
    k = int(weight.shape[1] * sparsity)
    if k == 0:
        return keep
    score = weight.detach().abs().float() * act_sq_sum.float().sqrt().unsqueeze(0)
    drop = torch.sort(score, dim=-1, stable=True).indices[:, :k]
    keep.scatter_(1, drop, False)
    return keep


def _sublayer(block: nn.Module, path: str) -> nn.Linear:
    mod = block
    for part in path.split("."):
        mod = getattr(mod, part)
    return mod


class _StopForward(Exception):
    pass


@torch.no_grad()
def _block_inputs(model, calib: torch.Tensor, device: str):
    """Run the embeddings once and capture each sample's input to block 0, plus the block's other arguments."""
    blocks = model.model.layers
    captured: list[torch.Tensor] = []
    kwargs_box: dict = {}

    def hook(_module, args, kwargs):
        captured.append(args[0] if args else kwargs["hidden_states"])
        kwargs_box.update({k: v for k, v in kwargs.items() if k != "hidden_states"})
        raise _StopForward

    handle = blocks[0].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        for sample in calib:
            try:
                model(sample.unsqueeze(0).to(device))
            except _StopForward:
                pass
    finally:
        handle.remove()
    return captured, kwargs_box


def _run_block(block, hidden: torch.Tensor, kwargs: dict) -> torch.Tensor:
    out = block(hidden, **kwargs)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def prune_wanda(
    model,
    calib: torch.Tensor,
    sparsity: float | Sequence[float],
    device: str,
    log=None,
) -> dict[str, torch.Tensor]:
    """Prune `model` in place with Wanda. `sparsity` is one value or one per block. Returns the keep-masks."""
    blocks = model.model.layers
    per_block = [sparsity] * len(blocks) if isinstance(sparsity, (int, float)) else list(sparsity)
    if len(per_block) != len(blocks):
        raise ValueError(f"{len(per_block)} sparsities for {len(blocks)} blocks")

    hiddens, kwargs = _block_inputs(model, calib, device)
    masks: dict[str, torch.Tensor] = {}
    for i, block in enumerate(blocks):
        if per_block[i] > 0:
            linears = {name: _sublayer(block, name) for name in LINEARS}
            sq_sums = {name: torch.zeros(lin.in_features, device=device) for name, lin in linears.items()}

            def make_hook(acc: torch.Tensor):
                def hook(_module, args, _output):
                    x = args[0].detach().float().reshape(-1, args[0].shape[-1])
                    acc.add_(x.pow(2).sum(dim=0))

                return hook

            handles = [lin.register_forward_hook(make_hook(sq_sums[n])) for n, lin in linears.items()]
            for h in hiddens:
                _run_block(block, h, kwargs)
            for handle in handles:
                handle.remove()
            for name, lin in linears.items():
                keep = row_mask(lin.weight, sq_sums[name], per_block[i])
                lin.weight.mul_(keep.to(lin.weight.dtype))
                masks[f"model.layers.{i}.{name}.weight"] = keep.cpu()
        hiddens = [_run_block(block, h, kwargs) for h in hiddens]
        if log:
            log(f"  block {i:2d}: sparsity {per_block[i]:.2f}")
    return masks


def linear_sparsity(model) -> float:
    """Fraction of zero weights across all block linear layers."""
    zeros = total = 0
    for block in model.model.layers:
        for name in LINEARS:
            w = _sublayer(block, name).weight
            zeros += int((w == 0).sum())
            total += w.numel()
    return zeros / total
