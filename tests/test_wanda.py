"""Tests for Wanda on a tiny random Qwen3 (offline, CPU)."""

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from pruneagent import wanda


@pytest.fixture
def tiny():
    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=128,
    )
    model = Qwen3ForCausalLM(cfg).eval()
    calib = torch.randint(0, 97, (4, 32))
    return model, calib


def test_row_mask_drops_exact_count_and_lowest_scores():
    w = torch.tensor([[1.0, -4.0, 2.0, 3.0], [5.0, 1.0, -1.0, 2.0]])
    act = torch.tensor([1.0, 1.0, 16.0, 1.0])  # sqrt -> [1, 1, 4, 1]: column 2 scores x4
    keep = wanda.row_mask(w, act, 0.5)
    assert keep.sum(dim=1).tolist() == [2, 2]
    # row 0 scores [1, 4, 8, 3] -> keep cols 1, 2; row 1 scores [5, 1, 4, 2] -> keep cols 0, 2
    assert keep.tolist() == [[False, True, True, False], [True, False, True, False]]


def test_zero_sparsity_keeps_everything():
    w = torch.randn(3, 8)
    assert wanda.row_mask(w, torch.ones(8), 0.0).all()


def test_uniform_half_sparsity(tiny):
    model, calib = tiny
    masks = wanda.prune_wanda(model, calib, 0.5, device="cpu")
    assert len(masks) == 3 * len(wanda.LINEARS)
    for keep in masks.values():
        assert (keep.sum(dim=1) == keep.shape[1] - int(keep.shape[1] * 0.5)).all()
    assert wanda.linear_sparsity(model) == pytest.approx(0.5)


def test_per_block_sparsity(tiny):
    model, calib = tiny
    wanda.prune_wanda(model, calib, [0.0, 0.25, 0.75], device="cpu")
    for i, s in enumerate([0.0, 0.25, 0.75]):
        w = model.model.layers[i].mlp.down_proj.weight
        assert (w == 0).float().mean().item() == pytest.approx(int(w.shape[1] * s) / w.shape[1])


def test_model_still_runs_after_pruning(tiny):
    model, calib = tiny
    wanda.prune_wanda(model, calib, 0.5, device="cpu")
    logits = model(calib[:1]).logits
    assert torch.isfinite(logits).all()


def test_wrong_sparsity_length_raises(tiny):
    model, calib = tiny
    with pytest.raises(ValueError):
        wanda.prune_wanda(model, calib, [0.5, 0.5], device="cpu")
