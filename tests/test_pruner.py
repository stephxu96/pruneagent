"""Pruner from a per-layer plan, plan rules and save / load, on a tiny random Qwen3 (offline, CPU)."""

import json

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from pruneagent import flap, loader, pruner
from pruneagent import plan as P
from pruneagent.wanda import LINEARS, _sublayer

N_LAYERS, WIDTH, GROUPS = 4, 256, 4


def tiny(tied=False, seed=0):
    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=97, hidden_size=64, intermediate_size=WIDTH, num_hidden_layers=N_LAYERS,
        num_attention_heads=8, num_key_value_heads=GROUPS, head_dim=16, max_position_embeddings=128,
        tie_word_embeddings=tied,
    )
    return Qwen3ForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def dense_and_scores():
    model = tiny()
    torch.manual_seed(1)
    scores = pruner.compute_scores(model, torch.randint(0, 97, (8, 16)), torch.randint(0, 97, (4, 32)), "cpu")
    return model, scores


def logits(model, ids):
    with torch.no_grad():
        return model(ids).logits


def mixed(cfg):
    steps = [{"layer": 0, "strategy": "narrow_mlp"}, {"layer": 0, "strategy": "narrow_mlp"},
             {"layer": 1, "strategy": "remove_attention_group"}, {"layer": 1, "strategy": "wanda_sparsity"},
             {"layer": 2, "strategy": "drop_layer"}, {"layer": 3, "strategy": "wanda_sparsity"},
             {"layer": 3, "strategy": "wanda_sparsity"}, {"layer": 3, "strategy": "remove_attention_group"}]
    return P.apply_steps(P.dense_plan(cfg), steps, cfg)


# ---------------------------------------------------------------- plan rules


def test_steps_qwen3_06b():
    cfg = {"num_hidden_layers": 28, "intermediate_size": 3072, "num_key_value_heads": 8}
    assert P.mlp_step(3072) == 320 and P.mlp_step(6144) == 640
    plan = P.step(P.dense_plan(cfg), 21, "narrow_mlp", cfg)
    assert plan[21]["mlp_keep"] == 2752  # §4.5 example
    plan = P.step(plan, 5, "wanda_sparsity", cfg)
    assert plan[5]["wanda_sparsity"] == 0.1


def test_validation_rejects_impossible_plans():
    cfg = tiny().config
    for bad, why in [({0: {"groups_keep": 0}}, "drop the layer"), ({0: {"mlp_keep": 100}}, "multiple of 64"),
                     ({0: {"mlp_keep": 0}}, "drop the layer"), ({1: {"groups_keep": GROUPS + 1}}, "outside"),
                     ({0: {"wanda_sparsity": 0.8}}, "outside"), ({9: {"dropped": True}}, "does not exist"),
                     ({i: {"dropped": True} for i in range(N_LAYERS)}, "every layer"),
                     ({0: {"mlp_kept": 64}}, "unknown fields")]:
        with pytest.raises(P.PlanError, match=why):
            P.validate(bad, cfg)
    one_group = P.validate({0: {"groups_keep": 1}}, cfg)
    with pytest.raises(P.PlanError, match="drop the layer"):
        P.step(one_group, 0, "remove_attention_group", cfg)
    seven = P.validate({0: {"wanda_sparsity": 0.7}}, cfg)
    with pytest.raises(P.PlanError):
        P.step(seven, 0, "wanda_sparsity", cfg)
    with pytest.raises(P.PlanError, match="already dropped"):
        P.step(P.validate({0: {"dropped": True}}, cfg), 0, "narrow_mlp", cfg)


def test_width_not_multiple_of_64_allowed_unpruned():
    cfg = {"num_hidden_layers": 1, "intermediate_size": 200, "num_key_value_heads": 2}
    assert P.validate({}, cfg)[0]["mlp_keep"] == 200
    assert P.step(P.dense_plan(cfg), 0, "narrow_mlp", cfg)[0]["mlp_keep"] == 128  # 200 -> 192 - 64


# ---------------------------------------------------------------- pruning


def test_dense_plan_changes_nothing(dense_and_scores):
    dense, scores = dense_and_scores
    ids = torch.randint(0, 97, (2, 12))
    out = pruner.prune(dense, {}, scores)
    assert torch.equal(logits(out, ids), logits(dense, ids))


def test_narrow_mlp_keeps_top_neurons_with_compensation(dense_and_scores):
    dense, scores = dense_and_scores
    out = pruner.prune(dense, {1: {"mlp_keep": 192}}, scores)
    keep = pruner.top_mask(scores.mlp[1], 192)
    assert keep.sum() == 192 and scores.mlp[1][keep].min() >= scores.mlp[1][~keep].max()
    ref = tiny()
    flap.remove_mlp_neurons(ref.model.layers[1], keep, scores.stats.mlp_mean[1])
    ids = torch.randint(0, 97, (2, 12))
    assert out.model.layers[1].mlp.down_proj.weight.shape == (64, 192)
    assert out.model.layers[1].mlp.down_proj.bias is not None
    assert out.model.layers[0].mlp.down_proj.bias is None
    assert torch.equal(logits(out, ids), logits(ref, ids))


def test_remove_group_drops_lowest_scored(dense_and_scores):
    dense, scores = dense_and_scores
    out = pruner.prune(dense, {2: {"groups_keep": 2}}, scores)
    a = out.model.layers[2].self_attn
    assert (a.q_proj.out_features, a.k_proj.out_features, a.o_proj.in_features) == (64, 32, 64)
    keep = pruner.top_mask(scores.attn[2], 2)
    lowest = torch.argsort(scores.attn[2])[:2]
    assert not keep[lowest].any()
    ref = tiny()
    flap.remove_attention_groups(ref.model.layers[2], keep, scores.stats.attn_mean[2])
    ids = torch.randint(0, 97, (1, 12))
    assert torch.equal(logits(out, ids), logits(ref, ids))


def test_drop_layer_equals_skipping_it(dense_and_scores):
    dense, scores = dense_and_scores
    out = pruner.prune(dense, {1: {"dropped": True}}, scores)
    assert len(out.model.layers) == N_LAYERS - 1 and out.config.num_hidden_layers == N_LAYERS - 1
    assert [b.self_attn.layer_idx for b in out.model.layers] == [0, 1, 2]
    assert len(out.config.layer_types) == N_LAYERS - 1
    ref = tiny()
    ref.model.layers[1].forward = lambda hidden_states, *a, **k: hidden_states
    ids = torch.randint(0, 97, (1, 12))
    assert torch.allclose(logits(out, ids), logits(ref, ids), atol=1e-6)
    assert dense.config.num_hidden_layers == N_LAYERS  # the unpruned copy is untouched


def test_wanda_sparsity_on_remaining_weights(dense_and_scores):
    dense, scores = dense_and_scores
    out = pruner.prune(dense, {0: {"wanda_sparsity": 0.3, "mlp_keep": 128, "groups_keep": 3}}, scores)
    block = out.model.layers[0]
    for name in LINEARS:
        w = _sublayer(block, name).weight
        assert ((w == 0).sum(dim=1) == int(w.shape[1] * 0.3)).all(), name
    # down_proj scored with the kept neurons' activation norms
    keep = pruner.top_mask(scores.mlp[0], 128)
    ref = tiny()
    flap.remove_mlp_neurons(ref.model.layers[0], keep, scores.stats.mlp_mean[0])
    w = ref.model.layers[0].mlp.down_proj.weight
    from pruneagent.wanda import row_mask
    expect = row_mask(w, scores.wanda_sq["mlp.down_proj"][0][keep], 0.3)
    assert torch.equal(block.mlp.down_proj.weight != 0, expect & (w != 0))
    assert (out.model.layers[1].mlp.up_proj.weight != 0).all()


def test_mixed_plan_is_deterministic(dense_and_scores):
    dense, scores = dense_and_scores
    plan = mixed(dense.config)
    a, b = pruner.prune(dense, plan, scores), pruner.prune(dense, plan, scores)
    ids = torch.randint(0, 97, (2, 12))
    assert torch.equal(logits(a, ids), logits(b, ids))
    counts = pruner.weight_counts(a)
    assert counts["block_weights"] < flap.block_params(dense) and counts["block_zeros"] > 0


def test_apply_plan_refuses_pruned_model(dense_and_scores):
    dense, scores = dense_and_scores
    out = pruner.prune(dense, {0: {"mlp_keep": 128}}, scores)
    with pytest.raises(ValueError, match="unpruned"):
        pruner.apply_plan(out, {}, scores)


def test_scores_roundtrip(dense_and_scores, tmp_path):
    _, scores = dense_and_scores
    scores.meta = {"model": "tiny"}
    pruner.save_scores(scores, tmp_path / "s.safetensors")
    back = pruner.load_scores(tmp_path / "s.safetensors")
    assert torch.equal(back.attn, scores.attn) and torch.equal(back.stats.mlp_mean, scores.stats.mlp_mean)
    assert all(torch.equal(back.wanda_sq[n], scores.wanda_sq[n]) for n in LINEARS)
    assert back.meta == {"model": "tiny"}


# ---------------------------------------------------------------- save and load


@pytest.mark.parametrize("tied", [False, True])
def test_save_load_identical_logits(tmp_path, tied):
    dense = tiny(tied=tied)
    scores = pruner.compute_scores(dense, torch.randint(0, 97, (8, 16)), torch.randint(0, 97, (4, 32)), "cpu")
    plan = mixed(dense.config)
    out = pruner.prune(dense, plan, scores)
    loader.save(out, tmp_path / "m", model_id="tiny", revision="abc")
    record = json.loads((tmp_path / "m" / loader.PLAN).read_text())
    assert record["kept_layers"] == [0, 1, 3] and record["plan"]["0"]["mlp_keep"] == 128
    back = loader.load(tmp_path / "m")
    ids = torch.randint(0, 97, (2, 12))
    assert torch.equal(logits(back, ids), logits(out, ids))
    assert sum(p.numel() for p in back.parameters()) == sum(p.numel() for p in out.parameters())
    if tied:
        assert back.lm_head.weight is back.model.embed_tokens.weight
    # a reloaded model saves again to identical weights
    loader.save(back, tmp_path / "again")
    assert json.loads((tmp_path / "again" / loader.PLAN).read_text())["weights"]["sha256"] == \
        record["weights"]["sha256"]


def test_load_rejects_mismatched_plan(tmp_path, dense_and_scores):
    dense, scores = dense_and_scores
    loader.save(pruner.prune(dense, {0: {"mlp_keep": 128}}, scores), tmp_path / "m")
    path = tmp_path / "m" / loader.PLAN
    record = json.loads(path.read_text())
    record["plan"]["0"]["mlp_keep"] = 192
    path.write_text(json.dumps(record))
    with pytest.raises(Exception, match="size mismatch|do not match"):
        loader.load(tmp_path / "m")


def test_generate_with_cache_after_drop(dense_and_scores, tmp_path):
    dense, scores = dense_and_scores
    out = pruner.prune(dense, {1: {"dropped": True}, 2: {"groups_keep": 3}}, scores)
    loader.save(out, tmp_path / "m")
    back = loader.load(tmp_path / "m")
    ids = torch.randint(0, 97, (1, 6))
    for model in (out, back):
        cached = model.generate(ids, max_new_tokens=8, do_sample=False, use_cache=True, pad_token_id=0)
        plain = model.generate(ids, max_new_tokens=8, do_sample=False, use_cache=False, pad_token_id=0)
        assert cached.shape == (1, 14) and torch.equal(cached, plain)
