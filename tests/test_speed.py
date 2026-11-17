"""Speed meter and device test on a tiny random Qwen3 (offline, CPU)."""

import json

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from pruneagent import pruner, speed


def tiny(tied=True):
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=8, num_key_value_heads=4, head_dim=16, max_position_embeddings=256,
                      tie_word_embeddings=tied)
    return Qwen3ForCausalLM(cfg).eval()


PROMPT = torch.randint(0, 97, (1, 300), generator=torch.Generator().manual_seed(1))


def test_measure_fields_and_exact_length():
    r = speed.measure(tiny(), "cpu", PROMPT, prompt_len=32, gen=10, repeats=3, warmup=1)
    assert r["generated_tokens"] == 10
    assert r["settings"] == {"prompt_len": 32, "batch": 1, "gen": 10, "decode_steps": 9, "repeats": 3,
                                  "warmup": 1, "seed": 0, "decoding": "greedy, key/value cache"}
    for key in ("prefill_tok_s", "decode_tok_s"):
        s = r[key]
        assert len(s["runs"]) == 3 and s["min"] <= s["median"] <= s["max"] and s["spread"] >= 0
    assert r["peak_memory_bytes"] is None and r["device"] == "cpu" and r["dtype"] == "float32"
    json.dumps(r)  # serializable


def test_generated_tokens_match_generate_and_never_stop_early():
    model = tiny()
    prompt = PROMPT[:, :20]
    ours = speed.generate_tokens(model, prompt, gen=12)
    assert ours.shape == (1, 12)
    ref = model.generate(prompt, max_new_tokens=12, min_new_tokens=12, do_sample=False, pad_token_id=0)
    assert torch.equal(ours, ref[:, 20:])
    # even when the third token is the end-of-sequence token, the meter generates all 12
    model.config.eos_token_id = model.generation_config.eos_token_id = int(ours[0, 2])
    assert speed.generate_tokens(model, prompt, gen=12).shape == (1, 12)


def test_measure_on_pruned_model_with_dropped_layer():
    model = tiny(tied=False)
    scores = pruner.compute_scores(model, torch.randint(0, 97, (4, 16)), torch.randint(0, 97, (2, 32)), "cpu")
    pruned = pruner.prune(model, {1: {"dropped": True}, 2: {"groups_keep": 2, "mlp_keep": 64}}, scores)
    r = speed.measure(pruned, "cpu", PROMPT, prompt_len=16, gen=5, repeats=2, warmup=0)
    assert r["generated_tokens"] == 5
    assert r["weight_bytes"] < speed.weight_bytes(model)


def test_weight_bytes_counts_tied_once():
    tied, untied = tiny(tied=True), tiny(tied=False)
    emb = 97 * 64 * 4
    assert speed.weight_bytes(untied) - speed.weight_bytes(tied) == emb


def test_short_prompt_rejected():
    with pytest.raises(ValueError):
        speed.measure(tiny(), "cpu", PROMPT[:, :8], prompt_len=16, gen=4, repeats=1, warmup=0)


def test_device_target_json():
    test = speed.device_test("cpu", copy_mb=4, matmul_n=64, iters=2, repeats=2, warmup=1)
    t = speed.device_target("laptop-cpu", "cpu", test)
    for key in ("name", "kind", "bandwidth_gbs", "compute_tflops", "measured_at"):
        assert key in t
    assert t["kind"] == "cpu" and t["bandwidth_gbs"] > 0 and t["compute_tflops"] > 0
    json.dumps(t)
