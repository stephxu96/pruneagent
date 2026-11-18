import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from pruneagent import flap


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


def logits(model, ids):
    with torch.no_grad():
        return model(ids).logits


class RefBiasGPT:
    """FLAP reference `BiasGPT.add_batch`, WIFV branch, verbatim apart from names (incl. the aliased old mean)."""

    def __init__(self, dim):
        self.nsamples = 0
        self.baseline_inp = torch.zeros(dim)
        self.fluc_inp = torch.zeros(dim)

    def add_batch(self, inp):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        batch_size = inp.shape[0]
        inp = inp.reshape((-1, inp.shape[-1])).t()
        old_baseline_inp = self.baseline_inp
        self.baseline_inp *= self.nsamples / (self.nsamples + batch_size)
        self.baseline_inp += torch.mean(inp, dim=1) / (self.nsamples + batch_size)
        if self.nsamples == 0:
            self.fluc_inp = 0
        else:
            self.fluc_inp *= (self.nsamples - 1) / (self.nsamples + batch_size - 1)
            self.fluc_inp += torch.sum(
                (inp - self.baseline_inp.unsqueeze(1)) * (inp - old_baseline_inp.unsqueeze(1)), dim=1
            ) / (self.nsamples + batch_size)
        self.nsamples += batch_size


def test_fluctuation_matches_reference():
    torch.manual_seed(1)
    ref, ours = RefBiasGPT(8), flap._Fluct(8, "cpu")
    for _ in range(5):
        x = torch.randn(1, 12, 8)
        ref.add_batch(x)
        ours.add(x)
    assert torch.allclose(ours.mean, ref.baseline_inp, atol=1e-6)
    assert torch.allclose(ours.fluct, ref.fluc_inp, atol=1e-5)


def test_group_weight_qwen3_06b():
    cfg = Qwen3Config(hidden_size=1024, num_attention_heads=16, num_key_value_heads=8, head_dim=128)

    class M:
        config = cfg

    assert flap.group_weight(M) == 256
    assert flap.group_shape(M) == (8, 256)


def test_removing_neurons_equals_zeroing_them():
    model, ref = tiny(), tiny()
    ids = torch.randint(0, 97, (2, 10))
    keep = torch.ones(128, dtype=torch.bool)
    keep[::3] = False
    with torch.no_grad():
        for b in ref.model.layers:
            b.mlp.gate_proj.weight[~keep] = 0
            b.mlp.up_proj.weight[~keep] = 0
    for b in model.model.layers:
        flap.remove_mlp_neurons(b, keep, None)
    assert model.model.layers[0].mlp.down_proj.weight.shape == (64, int(keep.sum()))
    assert torch.allclose(logits(model, ids), logits(ref, ids), atol=1e-5)


def test_removing_groups_equals_zeroing_them():
    model, ref = tiny(), tiny()
    ids = torch.randint(0, 97, (2, 10))
    keep = torch.tensor([False, True])  # group 0 = kv head 0 and query heads 0, 1
    with torch.no_grad():
        for b in ref.model.layers:
            b.self_attn.q_proj.weight[:32] = 0
            b.self_attn.k_proj.weight[:16] = 0
            b.self_attn.v_proj.weight[:16] = 0
    for b in model.model.layers:
        flap.remove_attention_groups(b, keep, None)
    a = model.model.layers[0].self_attn
    assert (a.q_proj.out_features, a.k_proj.out_features, a.o_proj.in_features) == (32, 16, 32)
    assert torch.allclose(logits(model, ids), logits(ref, ids), atol=1e-5)


def test_cannot_remove_every_group():
    with pytest.raises(ValueError):
        flap.remove_attention_groups(tiny().model.layers[0], torch.tensor([False, False]), None)


def test_bias_compensation_value():
    block = tiny().model.layers[0]
    w = block.mlp.down_proj.weight.detach().clone()
    mean = torch.randn(128)
    keep = torch.ones(128, dtype=torch.bool)
    keep[:10] = False
    flap.remove_mlp_neurons(block, keep, mean)
    assert torch.allclose(block.mlp.down_proj.bias, w[:, :10] @ mean[:10], atol=1e-5)


def test_allocate_counts_group_cost():
    attn = torch.tensor([[0.0, 5.0]])
    mlp = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    # descending: 5 (group, cost 2), 4, 3, 2, 1, 0 (group, cost 2); cumulative cost 2, 3, 4, ... target 8 * 0.5 = 4
    # lands on score 3; as in the reference, only scores strictly above the threshold are kept
    a, m = flap.allocate(attn, mlp, 0.5, weight=2.0)
    assert a.tolist() == [[False, True]]
    assert m.tolist() == [[False, False, False, True]]


def test_prune_flap_end_to_end():
    model = tiny()
    before = flap.block_params(model)
    calib = torch.randint(0, 97, (8, 16))
    out = flap.prune_flap(model, calib, 0.3, "cpu")
    after = flap.block_params(model)
    assert 0.15 < 1 - after / before < 0.45
    assert out["attn_keep"].shape == (3, 2) and out["mlp_keep"].shape == (3, 128)
    assert torch.isfinite(logits(model, torch.randint(0, 97, (1, 10)))).all()


def test_fluctuation_matches_reference_on_long_samples():
    """FLAP statistics on 2,048-token samples (the shared C4 set) agree with the reference update, which was
    published with 128-token samples; nothing in the update depends on the sample length."""
    torch.manual_seed(2)
    ref, ours = RefBiasGPT(16), flap._Fluct(16, "cpu")
    for _ in range(4):
        x = torch.randn(1, 2048, 16) * 3 + 1
        ref.add_batch(x)
        ours.add(x)
    assert torch.allclose(ours.mean, ref.baseline_inp, atol=1e-5)
    assert torch.allclose(ours.fluct, ref.fluc_inp, rtol=1e-4)


def test_collect_on_few_long_samples():
    model = tiny()
    stats = flap.collect(model, torch.randint(0, 97, (2, 128)), "cpu")
    for t in (stats.attn_fluct, stats.attn_mean, stats.mlp_fluct, stats.mlp_mean):
        assert t.shape[0] == 3 and torch.isfinite(t).all()
    assert (stats.mlp_fluct > 0).all()
