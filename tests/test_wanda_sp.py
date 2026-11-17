import pytest
import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from pruneagent import flap, wanda_sp


def tiny(seed=0):
    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
    )
    return Qwen3ForCausalLM(cfg).eval()


class RefWrappedGPT:
    """FLAP lib/layerwrapper.py `WrappedGPT` (commit 3bb57db, Apache-2.0), verbatim apart from unused fields."""

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.columns = layer.weight.data.shape[1]
        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples


def ref_masks(o_weight, o_scaler, d_weight, d_scaler, ratio, per_group):
    """FLAP lib/prune.py `prune_wanda_sp` scoring and threshold; 128 channels per head -> `per_group`, and
    num_heads -> number of key/value groups."""
    W_metric = torch.abs(o_weight) * torch.sqrt(o_scaler.reshape((1, -1)))
    W_metric = W_metric.mean(axis=0).reshape(-1, per_group).sum(dim=1)
    thresh = torch.sort(W_metric)[0][int(ratio * W_metric.numel())]
    attn = W_metric >= thresh
    W_metric = torch.abs(d_weight) * torch.sqrt(d_scaler.reshape((1, -1)))
    W_metric = W_metric.mean(axis=0)
    thresh = torch.sort(W_metric)[0][int(W_metric.numel() * ratio)]
    return attn, W_metric >= thresh


def test_scaler_row_matches_reference():
    torch.manual_seed(1)
    lin = nn.Linear(8, 4)
    ref, ours = RefWrappedGPT(lin), wanda_sp._ScalerRow(8, "cpu")
    for _ in range(5):
        x = torch.randn(1, 12, 8)
        ref.add_batch(x, None)
        ours.add(x)
    assert torch.allclose(ours.scaler_row, ref.scaler_row, atol=1e-6)


def test_scores_and_masks_match_reference():
    torch.manual_seed(2)
    o_w, d_w = torch.randn(64, 64), torch.randn(64, 128)
    o_s, d_s = torch.rand(64), torch.rand(128)
    for ratio in (0.0, 0.2, 0.5, 0.7):
        ref_a, ref_m = ref_masks(o_w, o_s, d_w, d_s, ratio, per_group=16)
        a = wanda_sp.keep_mask(wanda_sp.attention_scores(o_w, o_s, 16), ratio)
        m = wanda_sp.keep_mask(wanda_sp.mlp_scores(d_w, d_s), ratio)
        assert torch.equal(a, ref_a) and torch.equal(m, ref_m)


def test_threshold_behaviour():
    metric = torch.tensor([5.0, 1.0, 4.0, 2.0, 3.0])
    assert wanda_sp.keep_mask(metric, 0.0).all()
    assert wanda_sp.keep_mask(metric, 0.2).tolist() == [True, False, True, True, True]
    assert wanda_sp.keep_mask(metric, 0.5).tolist() == [True, False, True, False, True]  # int(2.5) = 2 go
    # ties at the threshold all stay, as in the reference
    assert wanda_sp.keep_mask(torch.tensor([1.0, 1.0, 1.0, 2.0]), 0.5).tolist() == [True] * 4


def test_end_to_end_shapes_and_matches_reference_loop():
    model, ref = tiny(), tiny()
    calib = torch.randint(0, 97, (6, 16))
    before = flap.block_params(model)
    out = wanda_sp.prune_wanda_sp(model, calib, 0.5, "cpu")
    assert out["attn_keep"].shape == (3, 2) and out["mlp_keep"].shape == (3, 128)
    assert out["attn_keep"].sum(dim=1).tolist() == [1, 1, 1]
    assert out["mlp_keep"].sum(dim=1).tolist() == [64, 64, 64]
    for b in model.model.layers:
        a = b.self_attn
        assert (a.q_proj.out_features, a.k_proj.out_features, a.o_proj.in_features) == (32, 16, 32)
        assert b.mlp.down_proj.weight.shape == (64, 64) and b.mlp.down_proj.bias is None
    assert 0.45 < 1 - flap.block_params(model) / before < 0.55
    assert torch.isfinite(model(torch.randint(0, 97, (1, 10))).logits).all()

    # the reference loop (WrappedGPT hooks, scoring, prune, rerun) on an identical model gives the same masks
    from pruneagent.wanda import _block_inputs, _run_block

    ref.config.use_cache = False
    with torch.no_grad():
        hiddens, kwargs = _block_inputs(ref, calib, "cpu")
        for i, layer in enumerate(ref.model.layers):
            subset = {"o": layer.self_attn.o_proj, "d": layer.mlp.down_proj}
            wrapped = {k: RefWrappedGPT(v) for k, v in subset.items()}
            handles = [subset[k].register_forward_hook(lambda _m, inp, o, w=wrapped[k]: w.add_batch(inp[0].data, o))
                       for k in subset]
            for h in hiddens:
                _run_block(layer, h, kwargs)
            for h in handles:
                h.remove()
            a, m = ref_masks(subset["o"].weight.data, wrapped["o"].scaler_row, subset["d"].weight.data,
                             wrapped["d"].scaler_row, 0.5, per_group=32)
            assert torch.equal(a, out["attn_keep"][i]) and torch.equal(m, out["mlp_keep"][i])
            flap.remove_attention_groups(layer, a, None)
            flap.remove_mlp_neurons(layer, m, None)
            hiddens = [_run_block(layer, h, kwargs) for h in hiddens]


def test_rejects_full_ratio():
    with pytest.raises(ValueError):
        wanda_sp.prune_wanda_sp(tiny(), torch.randint(0, 97, (2, 8)), 1.0, "cpu")
