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


class WindowBiasModel(torch.nn.Module):
    """Logits that depend on the window's first token, so each window has a different loss."""

    def __init__(self, vocab: int):
        super().__init__()
        self.vocab = vocab

    def forward(self, input_ids):
        x = input_ids[0]
        logits = torch.zeros(1, len(x), self.vocab)
        logits[0, :, int(x[0]) % self.vocab] = float(x[0]) / 3
        return SimpleNamespace(logits=logits)


def test_subset_indices_deterministic_even_and_nested():
    a = ev.subset_indices(128, 32)
    assert a == ev.subset_indices(128, 32) == list(range(0, 128, 4))
    assert set(a) <= set(ev.subset_indices(128, 64))
    assert ev.subset_indices(128, 128) == list(range(128))
    with pytest.raises(ValueError):
        ev.subset_indices(10, 11)


def test_subset_of_all_windows_equals_full_perplexity():
    ids = torch.randint(0, 40, (1, 16 * 8))
    m = WindowBiasModel(40)
    full = ev.perplexity(m, ids, seqlen=16, device="cpu")
    sub = ev.subset_perplexity(m, ids, n_windows=8, seqlen=16, device="cpu")
    assert sub.n_windows == 8 and sub.ppl == pytest.approx(full.ppl, rel=1e-6)


def test_subset_perplexity_matches_selected_windows():
    ids = torch.randint(0, 40, (1, 16 * 8 + 5))
    m = WindowBiasModel(40)
    full = ev.perplexity(m, ids, seqlen=16, device="cpu")
    sub = ev.subset_perplexity(m, ids, n_windows=2, seqlen=16, device="cpu")
    idx = ev.subset_indices(8, 2)
    assert idx == [0, 4]
    assert sub.window_losses == pytest.approx([full.window_losses[i] for i in idx])
    assert sub.ppl == pytest.approx(ev.ppl_from_losses(full.window_losses, idx), rel=1e-6)
    assert full.ppl == pytest.approx(ev.ppl_from_losses(full.window_losses), rel=1e-6)
