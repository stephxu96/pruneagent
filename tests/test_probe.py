"""Layer probes on a tiny random Qwen3 (offline, CPU)."""

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from pruneagent import plan as P
from pruneagent import probe as pr
from pruneagent import pruner

N_LAYERS = 4


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=256, num_hidden_layers=N_LAYERS,
                      num_attention_heads=8, num_key_value_heads=4, head_dim=16, max_position_embeddings=128)
    model = Qwen3ForCausalLM(cfg).eval()
    torch.manual_seed(1)
    scores = pruner.compute_scores(model, torch.randint(0, 97, (8, 16)), "cpu")
    acts = pr.capture(model, torch.randint(0, 97, (3, 24)), "cpu")
    return model, scores, acts


def test_capture_shapes(setup):
    _, _, acts = setup
    assert len(acts.inputs) == N_LAYERS + 1
    assert all(len(h) == 3 and h[0].shape == (1, 24, 64) for h in acts.inputs)


def test_table_shape(setup):
    model, scores, acts = setup
    table = pr.sensitivity_table(model, acts, scores, steps=(1, 2, 3))
    assert sorted(table) == list(range(N_LAYERS))
    for i, lp in table.items():
        assert lp["layer"] == i and 0 <= lp["block_influence"] <= 2
        cands = {(c["strategy"], c["steps"]): c for c in lp["candidates"]}
        assert set(cands) == {(s, k) for s in P.STRATEGIES for k in (1, 2, 3)}
        assert cands["drop_layer", 2]["local_error"] is None  # cannot drop twice
        for s in ("narrow_mlp", "remove_attention_group", "wanda_sparsity"):
            errs = [cands[s, k]["local_error"] for k in (1, 2, 3)]
            assert all(e > 0 for e in errs)
        assert cands["remove_attention_group", 1]["setting"]["groups_keep"] == 3


def test_drop_layer_is_identity_error(setup):
    model, scores, acts = setup
    i = 2
    x, y = acts.block_in(i), acts.block_out(i)
    expected = sum(float((a - b).pow(2).sum()) for a, b in zip(x, y)) / sum(float(b.pow(2).sum()) for b in y)
    got = pr.probe(model, i, ["drop_layer"], acts, scores)["candidates"][0]["local_error"]
    assert got == pytest.approx(expected, rel=1e-6)


def test_probe_matches_pruned_model(setup):
    """The probed block is the block the pruner builds for that plan."""
    model, scores, acts = setup
    i = 1
    plan = P.step(P.dense_plan(model.config), i, "narrow_mlp", model.config)
    pruned = pruner.prune(model, plan, scores)
    with torch.no_grad():
        y = [pr._run_block(pruned.model.layers[i], h, acts.kwargs) for h in acts.block_in(i)]
    got = pr.probe(model, i, ["narrow_mlp"], acts, scores)["candidates"][0]["local_error"]
    assert got == pytest.approx(pr.relative_error(y, acts.block_out(i)), rel=1e-6)


def test_deterministic_and_model_unchanged(setup):
    model, scores, _ = setup
    before = {k: v.clone() for k, v in model.state_dict().items()}
    calib = torch.randint(0, 97, (2, 16), generator=torch.Generator().manual_seed(5))
    a = pr.sensitivity_table(model, pr.capture(model, calib, "cpu"), scores)
    b = pr.sensitivity_table(model, pr.capture(model, calib.clone(), "cpu"), scores)
    assert a == b
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_spearman_and_compare():
    assert pr.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert pr.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert pr.spearman([1, 1, 2], [1, 1, 2]) == pytest.approx(1.0)
    table = {i: {"layer": i, "block_influence": 0.1 * i,
                 "candidates": [{"strategy": "drop_layer", "steps": 1, "setting": None, "local_error": e}]}
             for i, e in enumerate([0.3, 0.1, 0.2])}
    c = pr.compare(table, 10.0, {(0, "drop_layer"): 13.0, (1, "drop_layer"): 11.0, (2, "drop_layer"): 12.0})
    assert c["n"] == 3 and c["all"] == pytest.approx(1.0)
    assert c["per_strategy"]["drop_layer"]["spearman"] == pytest.approx(1.0)


def test_merge_probes():
    assert pr.merge_probes({1: "a", 2: "b"}, {2: "c"}) == {1: "a", 2: "c"}
    assert pr.merge_probes(None, {0: "x"}) == {0: "x"}
