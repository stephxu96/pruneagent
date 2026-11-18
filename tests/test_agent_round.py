"""Parsing and validation helpers of the hand-wired agent round (no Codex call)."""

import pytest

from pruneagent import agent_round as A
from pruneagent import plan as P

# Qwen3-0.6B shape, as a plain dict (no weights needed)
CFG = {"num_hidden_layers": 28, "intermediate_size": 3072, "num_key_value_heads": 8, "num_attention_heads": 16,
       "hidden_size": 1024, "head_dim": 128, "vocab_size": 151936, "tie_word_embeddings": True}

FAKE_REPLY = """Here is my plan.
```json
{"candidates": [
  {"choices": [{"layer": 24, "strategy": "drop_layer"}, {"layer": 10, "strategy": "narrow_mlp"},
               {"layer": 10, "strategy": "narrow_mlp"}], "rationale": "Drop one late layer, narrow a middle one."},
  {"choices": [{"layer": 5, "strategy": "remove_attention_group"}], "rationale": "Small attention cut."},
  {"choices": [{"layer": 3, "strategy": "drop_layer"}, {"layer": 4, "strategy": "drop_layer"}],
   "rationale": "Two drops."}
]}
```
tokens used
19,512
"""


def table():
    rows = {}
    for i in range(2):
        cands = [{"strategy": s, "steps": k, "setting": None, "local_error": None if s == "drop_layer" and k > 1
                  else 0.001 * (k + i)} for s in P.STRATEGIES for k in (1, 2, 3)]
        rows[str(i)] = {"layer": i, "block_influence": 0.1, "candidates": cands}
    return rows


def test_extract_json_from_fenced_reply_with_prose():
    obj = A.extract_json(FAKE_REPLY)
    assert len(obj["candidates"]) == 3


def test_extract_json_bare_and_skips_unrelated_objects():
    assert A.extract_json('{"candidates": []}') == {"candidates": []}
    assert A.extract_json('note {"a": 1} then {"candidates": [1]}') == {"candidates": [1]}
    with pytest.raises(ValueError, match="no JSON object"):
        A.extract_json('{"a": 1} and {broken')


def test_parse_reply_normalises_candidates():
    cands = A.parse_reply(FAKE_REPLY)
    assert [len(c["choices"]) for c in cands] == [3, 1, 2]
    assert cands[1]["rationale"] == "Small attention cut."
    bare = A.parse_reply('{"rationale": "shared", "candidates": [[{"layer": 1, "strategy": "narrow_mlp"}]]}')
    assert bare == [{"choices": [{"layer": 1, "strategy": "narrow_mlp"}], "rationale": "shared"}]
    with pytest.raises(TypeError):
        A.parse_reply('{"candidates": "none"}')


def test_tokens_used_from_codex_output():
    assert A.tokens_used(FAKE_REPLY) == 19512
    assert A.tokens_used("no count here") is None


def test_validate_candidates_applies_one_step_per_choice_and_rejects_two_drops():
    checks = A.validate_candidates(A.parse_reply(FAKE_REPLY), CFG)
    assert [c["valid"] for c in checks] == [True, True, False]
    plan = checks[0]["plan"]
    assert plan["24"]["dropped"] and plan["10"]["mlp_keep"] == 3072 - 2 * 320
    assert checks[1]["plan"]["5"]["groups_keep"] == 7
    assert "at most 1 per candidate" in checks[2]["reasons"][0]
    dense = A.E.weight_bytes(CFG, P.dense_plan(CFG))
    save = A.step_savings(CFG)
    # steps add up, except that the compensation bias is stored once per layer (not once per step)
    assert dense - checks[0]["weight_bytes"] == save["drop_layer"] + 2 * save["narrow_mlp"] + 2 * 1024


@pytest.mark.parametrize("choices, reason", [
    ([], "no choices"),
    ([{"layer": 1, "strategy": "narrow_mlp"}] * 3, "narrow_mlp 3 times"),
    ([{"layer": 28, "strategy": "narrow_mlp"}], "not a layer index"),
    ([{"layer": "1", "strategy": "narrow_mlp"}], "not a layer index"),
    ([{"layer": 1, "strategy": "quantize"}], "unknown strategy"),
    ([{"layer": 1, "strategy": "narrow_mlp", "amount": 2}], "expected"),
    ([{"layer": 2, "strategy": "narrow_mlp"}, {"layer": 2, "strategy": "drop_layer"}], "dropped and also"),
])
def test_check_candidate_rejections(choices, reason):
    plan, reasons = A.check_candidate(choices, CFG)
    assert plan is None and any(reason in r for r in reasons)


def test_check_candidate_run_limits_from_base_plan():
    base = P.normalize({i: {"dropped": True} for i in range(6)}, CFG)
    _, reasons = A.check_candidate([{"layer": 10, "strategy": "drop_layer"}], CFG, base, dropped_before=6)
    assert any("in the run" in r for r in reasons)
    narrow = P.normalize({3: {"mlp_keep": 832}}, CFG)  # one more step lands at 512 < 25% of 3,072
    _, reasons = A.check_candidate([{"layer": 3, "strategy": "narrow_mlp"}], CFG, narrow)
    assert any("below 25%" in r for r in reasons)
    last = P.normalize({3: {"groups_keep": 1}}, CFG)
    plan, reasons = A.check_candidate([{"layer": 3, "strategy": "remove_attention_group"}], CFG, last)
    assert plan is None and reasons


def test_more_than_three_candidates_rejected():
    one = {"choices": [{"layer": 1, "strategy": "narrow_mlp"}], "rationale": ""}
    checks = A.validate_candidates([one] * 4, CFG)
    assert [c["valid"] for c in checks] == [True, True, True, False]


def test_prompt_is_neutral_and_complete():
    limit = A.memory_limit(CFG)
    prompt = A.build_prompt(CFG, table(), limit)
    for s in P.STRATEGIES:
        assert f"- {s}:" in prompt
    assert "all layers unpruned" in prompt and "to remove: 119.210 MB" in prompt
    assert "0 | 0.100 | 1/2/3 | 1/2/3 | 1 | 1/2/3" in prompt
    assert '"candidates"' in prompt
