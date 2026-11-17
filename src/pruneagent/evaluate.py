"""Perplexity on cached WikiText-2 token streams, computed exactly as in the Wanda / SparseGPT reference.

The token stream is cut into non-overlapping `seqlen` windows (any remainder is dropped). For each
window the model's mean next-token cross-entropy is computed in fp32, multiplied by `seqlen`, and
perplexity = exp(sum / (n_windows * seqlen)). The reference multiplies by `seqlen` although each window
scores `seqlen - 1` tokens; kept as-is so numbers compare with published Wanda and FLAP results.
"""

from __future__ import annotations

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


@dataclass
class PplResult:
    ppl: float
    n_windows: int
    seqlen: int
    seconds: float


@torch.no_grad()
def perplexity(model, ids: torch.Tensor, seqlen: int = 2048, device: str | None = None) -> PplResult:
    """Wanda-style perplexity of `model` on a token stream. `model(input_ids).logits` must be [1, L, V]."""
    device = device or pick_device()
    wins = windows(ids, seqlen)
    nll_sum = 0.0
    synchronize(device)
    start = time.perf_counter()
    for w in wins:
        inp = w.unsqueeze(0).to(device)
        logits = model(inp).logits
        loss = F.cross_entropy(logits[0, :-1, :].float(), inp[0, 1:])
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss: the model overflowed in this dtype; try float32.")
        nll_sum += loss.item() * seqlen
    synchronize(device)
    seconds = time.perf_counter() - start
    ppl = float(torch.exp(torch.tensor(nll_sum / (len(wins) * seqlen))))
    return PplResult(ppl=ppl, n_windows=len(wins), seqlen=seqlen, seconds=seconds)


DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def load_model(model_id: str, revision: str | None, dtype: str, device: str):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=DTYPES[dtype])
    model.config.use_cache = False
    return model.to(device).eval()
