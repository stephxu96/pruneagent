"""Tests for the perplexity evaluator (offline, CPU, tiny fake models)."""

import math
from types import SimpleNamespace

import pytest
import torch

from pruneagent import evaluate as ev


class UniformModel(torch.nn.Module):
    """Equal logits for every token: cross-entropy = log(V), so perplexity = V."""

    def __init__(self, vocab: int):
        super().__init__()
        self.vocab = vocab

    def forward(self, input_ids):
        return SimpleNamespace(logits=torch.zeros(*input_ids.shape, self.vocab))


class OracleModel(torch.nn.Module):
    """Puts almost all probability on the true next token: perplexity close to 1."""

    def __init__(self, vocab: int):
        super().__init__()
        self.vocab = vocab

    def forward(self, input_ids):
        x = input_ids[0]
        logits = torch.full((1, len(x), self.vocab), -1e4)
        logits[0, torch.arange(len(x) - 1), x[1:]] = 0.0
        return SimpleNamespace(logits=logits)


def test_windows_drop_remainder():
    w = ev.windows(torch.arange(10).unsqueeze(0), 4)
    assert w.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_uniform_model_perplexity_is_vocab_size():
    ids = torch.randint(0, 50, (1, 64))
    r = ev.perplexity(UniformModel(50), ids, seqlen=16, device="cpu")
    assert r.n_windows == 4 and r.seqlen == 16
    assert r.ppl == pytest.approx(50, rel=1e-5)


def test_oracle_model_perplexity_near_one():
    ids = torch.randint(0, 30, (1, 40))
    r = ev.perplexity(OracleModel(30), ids, seqlen=10, device="cpu")
    assert r.ppl == pytest.approx(1.0, abs=1e-3)


def test_reference_formula_scales_by_seqlen():
    # Each window scores seqlen-1 tokens but is weighted by seqlen, as in the Wanda reference.
    ids = torch.randint(0, 7, (1, 21))
    r = ev.perplexity(UniformModel(7), ids, seqlen=7, device="cpu")
    assert math.log(r.ppl) == pytest.approx(math.log(7), rel=1e-6)


def test_non_finite_loss_raises():
    class NanModel(UniformModel):
        def forward(self, input_ids):
            return SimpleNamespace(logits=torch.full((*input_ids.shape, self.vocab), float("nan")))

    with pytest.raises(FloatingPointError):
        ev.perplexity(NanModel(5), torch.zeros(1, 10, dtype=torch.long), seqlen=5, device="cpu")
