"""Bring your own model (FR17): the pruner, loader and size estimate handle every supported family exactly, on tiny
random models (offline, CPU). A family is listed in `models.FAMILIES` only when it passes here."""

import pytest
import torch
import transformers as T
from test_pruner import mixed

from pruneagent import estimate as E
from pruneagent import loader, models, pruner

KW = {"vocab_size": 97, "hidden_size": 64, "intermediate_size": 256, "num_hidden_layers": 4,
      "num_attention_heads": 8, "num_key_value_heads": 4, "max_position_embeddings": 128}
BUILD = {
    "qwen3": lambda: T.Qwen3ForCausalLM(T.Qwen3Config(head_dim=16, **KW)),
    "qwen2": lambda: T.Qwen2ForCausalLM(T.Qwen2Config(**KW)),
    "llama": lambda: T.LlamaForCausalLM(T.LlamaConfig(**KW)),
    "mistral": lambda: T.MistralForCausalLM(T.MistralConfig(**KW)),
}


def test_every_supported_family_is_tested_here():
    assert set(BUILD) == set(models.FAMILIES)


@pytest.mark.parametrize("family", sorted(BUILD))
def test_family_prunes_generates_reloads_and_matches_the_estimate(family, tmp_path):
    torch.manual_seed(0)
    model = BUILD[family]().eval()
    ids = torch.randint(0, 97, (8, 16))
    plan = mixed(model.config)  # every strategy, including a removed layer
    pruned = pruner.prune(model, plan, pruner.compute_scores(model, ids, "cpu"))
    with torch.no_grad():
        out = pruned(ids[:2]).logits
    assert torch.isfinite(out).all()
    assert pruned.generate(ids[:1, :4], max_new_tokens=3, do_sample=False).shape[1] == 7
    assert E.weight_bytes(model.config, plan) == sum(p.numel() for p in pruned.parameters()) * 2
    loader.save(pruned, tmp_path, model_id=family)
    with torch.no_grad():
        assert torch.allclose(loader.load(tmp_path)(ids[:2]).logits, out, atol=1e-4)


def test_model_names_are_checked_before_any_download():
    assert models.valid_id("HuggingFaceTB/SmolLM2-360M") and models.valid_id("Qwen/Qwen3-0.6B-Base")
    for bad in ("", "x", "a/b/c", "a/b; ls", "../a/b", "-a/b"):
        assert not models.valid_id(bad)
    with pytest.raises(models.ModelError):
        models.check("not a name", 16)


def test_prepare_time_matches_the_measured_1_7b():
    lo, hi = models.prepare_minutes(1.72, 28)  # measured: 2 h 55 min (175 min) on this Mac
    assert lo < 175 < hi
