"""Device estimator: weight bytes against the saved file, monotonic in pruning, Wanda saves nothing; device check."""

import json
import struct

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from pruneagent import estimate as E
from pruneagent import loader, pruner
from pruneagent import plan as P

N_LAYERS, WIDTH, GROUPS = 4, 256, 4


def tiny(tied=False, seed=0):
    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=512, hidden_size=128, intermediate_size=WIDTH, num_hidden_layers=N_LAYERS,
        num_attention_heads=8, num_key_value_heads=GROUPS, head_dim=16, max_position_embeddings=128,
        tie_word_embeddings=tied,
    )
    return Qwen3ForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def models():
    out = {}
    for tied in (False, True):
        m = tiny(tied)
        torch.manual_seed(1)
        out[tied] = m, pruner.compute_scores(m, torch.randint(0, 512, (4, 16)), torch.randint(0, 512, (2, 32)), "cpu")
    return out


def mixed(cfg):
    steps = [{"layer": 0, "strategy": "narrow_mlp"}, {"layer": 0, "strategy": "narrow_mlp"},
             {"layer": 1, "strategy": "remove_attention_group"}, {"layer": 1, "strategy": "wanda_sparsity"},
             {"layer": 2, "strategy": "drop_layer"}, {"layer": 3, "strategy": "wanda_sparsity"},
             {"layer": 3, "strategy": "remove_attention_group"}]
    return P.apply_steps(P.dense_plan(cfg), steps, cfg)


PLANS = {
    "dense": lambda cfg: {},
    "narrow_mlp": lambda cfg: {1: {"mlp_keep": 192}},
    "remove_attention_group": lambda cfg: {2: {"groups_keep": 2}},
    "drop_layer": lambda cfg: {1: {"dropped": True}},
    "wanda_sparsity": lambda cfg: {0: {"wanda_sparsity": 0.5}},
    "mixed": mixed,
}


def data_bytes(path) -> int:
    """Tensor data in a safetensors file: file size minus the 8-byte length and the JSON header."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
    return path.stat().st_size - 8 - n


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("name", list(PLANS))
def test_weight_bytes_match_saved_file(models, tmp_path, name, tied):
    dense, scores = models[tied]
    plan = PLANS[name](dense.config)
    out = loader.save(pruner.prune(dense, plan, scores), tmp_path / name)
    f = out / loader.WEIGHTS
    est = E.weight_bytes(dense.config, plan, dtype_bytes=4)
    assert est == data_bytes(f)  # exact on the tensor data
    assert abs(est - f.stat().st_size) / f.stat().st_size < 0.01  # the done-when: file size within 1%


def test_counts_match_rebuilt_model(models, tmp_path):
    dense, scores = models[True]
    plan = mixed(dense.config)
    back = loader.load(loader.save(pruner.prune(dense, plan, scores), tmp_path / "m"))
    stored = sum(t.numel() for t in back.state_dict().values()) - back.lm_head.weight.numel()  # tied: once
    assert E.weight_bytes(dense.config, plan, dtype_bytes=4) == stored * 4


def test_monotonic_in_pruning():
    cfg = tiny().config
    plan, prev = P.dense_plan(cfg), E.estimate(cfg, {}, E.PRESETS["t4"])
    for layer, strategy in [(0, "narrow_mlp"), (1, "remove_attention_group"), (0, "narrow_mlp"),
                            (2, "drop_layer"), (3, "remove_attention_group")]:
        plan = P.step(plan, layer, strategy, cfg)
        e = E.estimate(cfg, plan, E.PRESETS["t4"])
        assert e.weight_bytes < prev.weight_bytes
        assert e.decode_bytes_per_token < prev.decode_bytes_per_token
        assert e.prefill_flops < prev.prefill_flops
        assert e.est_decode_tok_s > prev.est_decode_tok_s
        prev = e
    # removing a group shrinks the key/value cache; narrowing the MLP does not
    assert E.kv_cache_bytes(cfg, {0: {"groups_keep": 3}}) < E.kv_cache_bytes(cfg, {})
    assert E.kv_cache_bytes(cfg, {0: {"mlp_keep": 64}}) == E.kv_cache_bytes(cfg, {})


def test_wanda_sparsity_saves_nothing():
    cfg = tiny().config
    sparse = {i: {"wanda_sparsity": 0.7} for i in range(N_LAYERS)}
    assert E.estimate(cfg, sparse, E.PRESETS["m4"]) == E.estimate(cfg, {}, E.PRESETS["m4"])


def test_qwen3_06b_dense_numbers():
    cfg = {"vocab_size": 151936, "hidden_size": 1024, "intermediate_size": 3072, "num_hidden_layers": 28,
           "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128, "tie_word_embeddings": True}
    # 596,049,920 parameters stored (lm_head tied), fp16
    assert E.weight_bytes(cfg, {}) == 596_049_920 * 2
    assert E.kv_cache_bytes(cfg, {}, context=512) == 28 * 2 * 8 * 128 * 512 * 2


def test_check_reports_gaps():
    cfg = tiny().config
    e = E.estimate(cfg, {}, E.PRESETS["t4"])
    roomy = E.DeviceTarget("roomy", memory_limit_gb=1, bandwidth_gb_s=100, compute_tflops=10,
                           min_decode_tok_s=1.0)
    c = E.check(roomy, cfg, {})
    assert c.meets_target and c.memory_gap_gb == 0 and c.speed_gap_tok_s == 0 and c.estimate.meets_target
    tight = E.DeviceTarget("tight", memory_limit_gb=e.memory_bytes / 2 / E.GB, bandwidth_gb_s=1, compute_tflops=1,
                           backend="cpu", min_decode_tok_s=1e6)
    c = E.check(tight, cfg, {})
    assert not c.fits_memory and not c.meets_speed and not c.meets_target
    assert c.memory_gap_gb == pytest.approx(e.memory_bytes / 2 / E.GB)
    assert c.speed_gap_tok_s == pytest.approx(1e6 - c.estimate.est_decode_tok_s)
    no_speed = E.DeviceTarget("no speed", memory_limit_gb=1, bandwidth_gb_s=1, compute_tflops=1)
    assert E.check(no_speed, cfg, {}).meets_speed
    json.dumps(c.to_dict())


def test_decode_speed_is_bandwidth_over_bytes():
    cfg = tiny().config
    t = E.DeviceTarget("x", memory_limit_gb=1, bandwidth_gb_s=100, compute_tflops=1000)
    e = E.estimate(cfg, {}, t)
    assert e.est_decode_tok_s == pytest.approx(100e9 / e.decode_bytes_per_token)
    fast_memory = E.DeviceTarget("y", memory_limit_gb=1, bandwidth_gb_s=1e9, compute_tflops=1)
    assert E.estimate(cfg, {}, fast_memory).est_prefill_s == pytest.approx(e.prefill_flops / 1e12)


def test_targets(tmp_path):
    assert E.load_target("t4").bandwidth_gb_s == 320
    p = tmp_path / "phone.json"
    p.write_text(json.dumps({"name": "phone", "memory_limit_gb": 2.5, "bandwidth_gb_s": 55, "compute_tflops": 2,
                             "min_decode_tok_s": 20}))
    assert E.load_target(str(p)).min_decode_tok_s == 20
    with pytest.raises(ValueError):
        E.load_target("nope")
    with pytest.raises(ValueError):
        E.DeviceTarget("bad", memory_limit_gb=1, bandwidth_gb_s=1, compute_tflops=1, backend="npu")
