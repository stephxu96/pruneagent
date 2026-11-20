"""Perplexity on cached WikiText-2 token streams, computed exactly as in the Wanda / SparseGPT reference.

The token stream is cut into non-overlapping `seqlen` windows (any remainder is dropped). For each
window the model's mean next-token cross-entropy is computed in fp32, multiplied by `seqlen`, and
perplexity = exp(sum / (n_windows * seqlen)). The reference multiplies by `seqlen` although each window
scores `seqlen - 1` tokens; kept as-is so numbers compare with published Wanda and FLAP results.

The search gate (FR6, §4.4) scores candidates on a fixed subset of the validation windows: `subset_indices`
picks evenly spaced windows (window i * n_total // n for i < n; for 32 of 128, every 4th window from 0), so
the choice is deterministic, spread across the whole split, and nested (the 32-window subset lies inside the
64-window one). `subset_perplexity` is the gate's measurement; it is the same computation on those windows.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def windows(ids: torch.Tensor, seqlen: int) -> torch.Tensor:
    """[1, N] or [N] token stream → [N // seqlen, seqlen] non-overlapping windows."""
    flat = ids.reshape(-1)
    n = flat.numel() // seqlen
    return flat[: n * seqlen].reshape(n, seqlen)


GATE_WINDOWS = 32  # validation windows the gate checks each candidate on (ARCHITECTURE.md FR6, A9)


def subset_indices(n_total: int, n: int) -> list[int]:
    """`n` evenly spaced window indices out of `n_total`, starting at 0. Same input, same subset."""
    if not 1 <= n <= n_total:
        raise ValueError(f"subset of {n} windows out of {n_total}")
    return [i * n_total // n for i in range(n)]


def ppl_from_losses(window_losses: list[float], idx: list[int] | None = None) -> float:
    """Perplexity from per-window mean losses (all windows, or those in `idx`), as `perplexity` computes it."""
    sel = window_losses if idx is None else [window_losses[i] for i in idx]
    return math.exp(sum(sel) / len(sel))


def window_loss(logits: torch.Tensor, inp: torch.Tensor) -> float:
    """Mean next-token loss of one window ([1, L, V] logits for [1, L] tokens), in fp32."""
    loss = F.cross_entropy(logits[0, :-1, :].float(), inp[0, 1:])
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite loss: the model overflowed in this dtype; try float32.")
    return loss.item()


def ppl_of(losses: list[float], seqlen: int) -> float:
    """Perplexity from per-window losses, computed as the reference does (each loss times `seqlen`)."""
    nll_sum = 0.0
    for loss in losses:
        nll_sum += loss * seqlen
    return float(torch.exp(torch.tensor(nll_sum / (len(losses) * seqlen))))


@dataclass
class PplResult:
    ppl: float
    n_windows: int
    seqlen: int
    seconds: float
    window_losses: list[float]  # mean next-token loss of each evaluated window, in order


@torch.no_grad()
def perplexity(model, ids: torch.Tensor, seqlen: int = 2048, device: str | None = None,
               window_idx: list[int] | None = None) -> PplResult:
    """Wanda-style perplexity of `model` on a token stream (only the windows in `window_idx`, if given).
    `model(input_ids).logits` must be [1, L, V]."""
    device = device or pick_device()
    wins = windows(ids, seqlen)
    if window_idx is not None:
        wins = wins[torch.as_tensor(window_idx, dtype=torch.long)]
    losses = []
    synchronize(device)
    start = time.perf_counter()
    for w in wins:
        inp = w.unsqueeze(0).to(device)
        losses.append(window_loss(model(inp).logits, inp))
    synchronize(device)
    seconds = time.perf_counter() - start
    return PplResult(ppl=ppl_of(losses, seqlen), n_windows=len(wins), seqlen=seqlen, seconds=seconds,
                     window_losses=losses)


def subset_perplexity(model, ids: torch.Tensor, n_windows: int = GATE_WINDOWS, seqlen: int = 2048,
                      device: str | None = None) -> PplResult:
    """The gate's perplexity: `perplexity` on the fixed evenly spaced subset of `n_windows` windows."""
    n_total = ids.numel() // seqlen
    return perplexity(model, ids, seqlen, device, window_idx=subset_indices(n_total, n_windows))


DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def load_model(model_id: str, revision: str | None, dtype: str, device: str):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=DTYPES[dtype])
    model.config.use_cache = False
    return model.to(device).eval()


def free_memory(device: str) -> None:
    """Give a dropped model's memory back. `del model` alone keeps it: Python frees the model only on its next
    collection and the GPU keeps the freed blocks cached, so building the next model in a check stacked on top of the
    last one (1.7B quiz check: ~12 GB held, the Mac swapped and the second model took 4x as long)."""
    import gc

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
